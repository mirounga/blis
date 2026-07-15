/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#include <arm_sme.h>

// Streaming-SVE packm kernels for the SME2 gemm microkernels.
//
// Apple cores have no non-streaming SVE, so the vectorized path runs in
// streaming mode (one streaming-mode switch per micro-panel; a panel is
// cdim x n with n up to KC, so the switch cost is well amortized). ZA is
// not used, hence no __arm_new("za").
//
// The kernels are cdim-driven: the same routine packs A micro-panels
// (cdim_max = MR) and B micro-panels (cdim_max = NR) of any width, with
// zero-padding of rows cdim..cdim_max and columns n..n_max, matching the
// reference packm semantics. Only the common case (inca == 1, no broadcast)
// is vectorized; other cases fall back to scalar code.

// -- float --------------------------------------------------------------------

__arm_locally_streaming
static void bli_spackm_armsme_int_2vlxk_body
     (
       dim_t        cdim,
       dim_t        cdim_max,
       dim_t        n,
       dim_t        n_max,
       float        kappa,
       const float* a, inc_t lda,
       float*       p, inc_t ldp
     )
{
	const uint64_t    vl     = svcntw();
	const dim_t       nv     = ( cdim_max + ( dim_t )vl - 1 ) / ( dim_t )vl;
	const svbool_t    ptrue  = svptrue_b32();
	const svfloat32_t vkappa = svdup_f32( kappa );
	const svfloat32_t vzero  = svdup_f32( 0.0f );

	for ( dim_t j = 0; j < n; ++j )
	{
		const float* aj = a + j*lda;
		float*       pj = p + j*ldp;

		for ( dim_t v = 0; v < nv; ++v )
		{
			// Inactive (predicated-off) lanes of svld1 read as zero, so
			// rows cdim..cdim_max are zero-filled by the wider store.
			const svbool_t pld = svwhilelt_b32( ( uint64_t )( v*vl ), ( uint64_t )cdim );
			const svbool_t pst = svwhilelt_b32( ( uint64_t )( v*vl ), ( uint64_t )cdim_max );

			svfloat32_t x = svld1_f32( pld, aj + v*vl );
			x = svmul_f32_x( ptrue, x, vkappa );
			svst1_f32( pst, pj + v*vl, x );
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		float* pj = p + j*ldp;

		for ( dim_t v = 0; v < nv; ++v )
		{
			const svbool_t pst = svwhilelt_b32( ( uint64_t )( v*vl ), ( uint64_t )cdim_max );
			svst1_f32( pst, pj + v*vl, vzero );
		}
	}
}

// Transpose path: non-unit inca with unit lda (lda == 1), i.e. source row i is
// contiguous over k at a + i*inca. This is the packing "transpose" case (e.g.
// the B micro-panel of a column-major gemm) that the scalar fallback used to
// handle. ZA is used as a transpose engine: hor-load source rows into ZA tile
// slices, ver-read columns back out (= transpose), scale by kappa, and store
// k-major. ZA is free during packing (the gemm ukr uses ZA in a later, separate
// call), so this body owns ZA via __arm_new("za").
__arm_new("za") __arm_locally_streaming
static void bli_spackm_armsme_int_2vlxk_trans
     (
       dim_t        cdim,
       dim_t        cdim_max,
       dim_t        n,
       dim_t        n_max,
       float        kappa,
       const float* a, inc_t inca,
       float*       p, inc_t ldp
     )
{
	const uint64_t    vl     = svcntw();
	const svbool_t    ptrue  = svptrue_b32();
	const svfloat32_t vkappa = svdup_f32( kappa );
	const svfloat32_t vzero  = svdup_f32( 0.0f );

	for ( dim_t kb = 0; kb < n; kb += ( dim_t )vl )
	{
		const dim_t    kk = ( n - kb < ( dim_t )vl ) ? ( n - kb ) : ( dim_t )vl;
		const svbool_t pk = svwhilelt_b32( ( uint64_t )0, ( uint64_t )kk );

		// Cover cdim_max rows (not just cdim) so multi-VL row padding is written.
		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t sr = ( rb < cdim )
			                 ? ( ( cdim - rb < ( dim_t )vl ) ? ( cdim - rb ) : ( dim_t )vl )
			                 : 0;

			// Load up to VL source rows as horizontal ZA slices (kk k-values each).
			for ( dim_t s = 0; s < sr; ++s )
				svld1_hor_za32( 0, ( uint32_t )s, pk, a + ( rb + s )*inca + kb );

			// pr masks rows rb..cdim: inactive lanes (padding rows, and stale/
			// unloaded slices when sr < VL) merge from vzero, so no svzero_za is
			// needed. pst extends the store to cdim_max to write the row padding.
			const svbool_t pr  = svwhilelt_b32( ( uint64_t )rb, ( uint64_t )cdim );
			const svbool_t pst = svwhilelt_b32( ( uint64_t )rb, ( uint64_t )cdim_max );

			for ( dim_t q = 0; q < kk; ++q )
			{
				svfloat32_t x = svread_ver_za32_f32_m( vzero, pr, 0, ( uint32_t )q );
				x = svmul_f32_x( ptrue, x, vkappa );
				svst1_f32( pst, p + ( kb + q )*ldp + rb, x );
			}
		}
	}

	// Column padding: k-slices n..n_max are all zero.
	for ( dim_t j = n; j < n_max; ++j )
	{
		float* pj = p + j*ldp;
		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const svbool_t pst = svwhilelt_b32( ( uint64_t )rb, ( uint64_t )cdim_max );
			svst1_f32( pst, pj + rb, vzero );
		}
	}
}

void bli_spackm_armsme_int_2vlxk
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n,
             dim_t   n_max,
       const void*   kappa,
       const void*   a, inc_t inca, inc_t lda,
             void*   p,             inc_t ldp,
       const void*   params,
       const cntx_t* cntx
     )
{
	const float  kappa_cast = *( const float* )kappa;
	const float* alpha1     = a;
	      float* pi1        = p;

	if ( inca == 1 && cdim_bcast == 1 )
	{
		bli_spackm_armsme_int_2vlxk_body
		(
		  cdim, cdim_max, n, n_max,
		  kappa_cast, alpha1, lda, pi1, ldp
		);
	}
	else if ( lda == 1 && cdim_bcast == 1 )
	{
		// inca != 1 here (inca == 1 handled above): pure transpose.
		bli_spackm_armsme_int_2vlxk_trans
		(
		  cdim, cdim_max, n, n_max,
		  kappa_cast, alpha1, inca, pi1, ldp
		);
	}
	else
	{
		for ( dim_t j = 0; j < n; ++j )
		{
			for ( dim_t i = 0; i < cdim; ++i )
			for ( dim_t d = 0; d < cdim_bcast; ++d )
				pi1[ j*ldp + i*cdim_bcast + d ] = kappa_cast * alpha1[ j*lda + i*inca ];

			for ( dim_t i = cdim*cdim_bcast; i < cdim_max*cdim_bcast; ++i )
				pi1[ j*ldp + i ] = 0.0f;
		}

		for ( dim_t j = n; j < n_max; ++j )
			for ( dim_t i = 0; i < cdim_max*cdim_bcast; ++i )
				pi1[ j*ldp + i ] = 0.0f;
	}
}

// -- double -------------------------------------------------------------------

__arm_locally_streaming
static void bli_dpackm_armsme_int_2vlxk_body
     (
       dim_t         cdim,
       dim_t         cdim_max,
       dim_t         n,
       dim_t         n_max,
       double        kappa,
       const double* a, inc_t lda,
       double*       p, inc_t ldp
     )
{
	const uint64_t    vl     = svcntd();
	const dim_t       nv     = ( cdim_max + ( dim_t )vl - 1 ) / ( dim_t )vl;
	const svbool_t    ptrue  = svptrue_b64();
	const svfloat64_t vkappa = svdup_f64( kappa );
	const svfloat64_t vzero  = svdup_f64( 0.0 );

	for ( dim_t j = 0; j < n; ++j )
	{
		const double* aj = a + j*lda;
		double*       pj = p + j*ldp;

		for ( dim_t v = 0; v < nv; ++v )
		{
			const svbool_t pld = svwhilelt_b64( ( uint64_t )( v*vl ), ( uint64_t )cdim );
			const svbool_t pst = svwhilelt_b64( ( uint64_t )( v*vl ), ( uint64_t )cdim_max );

			svfloat64_t x = svld1_f64( pld, aj + v*vl );
			x = svmul_f64_x( ptrue, x, vkappa );
			svst1_f64( pst, pj + v*vl, x );
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		double* pj = p + j*ldp;

		for ( dim_t v = 0; v < nv; ++v )
		{
			const svbool_t pst = svwhilelt_b64( ( uint64_t )( v*vl ), ( uint64_t )cdim_max );
			svst1_f64( pst, pj + v*vl, vzero );
		}
	}
}

// Transpose path (lda == 1, inca != 1): ZA64 as a transpose engine. See the
// float bli_spackm_armsme_int_2vlxk_trans for the full explanation.
__arm_new("za") __arm_locally_streaming
static void bli_dpackm_armsme_int_2vlxk_trans
     (
       dim_t         cdim,
       dim_t         cdim_max,
       dim_t         n,
       dim_t         n_max,
       double        kappa,
       const double* a, inc_t inca,
       double*       p, inc_t ldp
     )
{
	const uint64_t    vl     = svcntd();
	const svbool_t    ptrue  = svptrue_b64();
	const svfloat64_t vkappa = svdup_f64( kappa );
	const svfloat64_t vzero  = svdup_f64( 0.0 );

	for ( dim_t kb = 0; kb < n; kb += ( dim_t )vl )
	{
		const dim_t    kk = ( n - kb < ( dim_t )vl ) ? ( n - kb ) : ( dim_t )vl;
		const svbool_t pk = svwhilelt_b64( ( uint64_t )0, ( uint64_t )kk );

		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t sr = ( rb < cdim )
			                 ? ( ( cdim - rb < ( dim_t )vl ) ? ( cdim - rb ) : ( dim_t )vl )
			                 : 0;

			for ( dim_t s = 0; s < sr; ++s )
				svld1_hor_za64( 0, ( uint32_t )s, pk, a + ( rb + s )*inca + kb );

			const svbool_t pr  = svwhilelt_b64( ( uint64_t )rb, ( uint64_t )cdim );
			const svbool_t pst = svwhilelt_b64( ( uint64_t )rb, ( uint64_t )cdim_max );

			for ( dim_t q = 0; q < kk; ++q )
			{
				svfloat64_t x = svread_ver_za64_f64_m( vzero, pr, 0, ( uint32_t )q );
				x = svmul_f64_x( ptrue, x, vkappa );
				svst1_f64( pst, p + ( kb + q )*ldp + rb, x );
			}
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		double* pj = p + j*ldp;
		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const svbool_t pst = svwhilelt_b64( ( uint64_t )rb, ( uint64_t )cdim_max );
			svst1_f64( pst, pj + rb, vzero );
		}
	}
}

void bli_dpackm_armsme_int_2vlxk
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n,
             dim_t   n_max,
       const void*   kappa,
       const void*   a, inc_t inca, inc_t lda,
             void*   p,             inc_t ldp,
       const void*   params,
       const cntx_t* cntx
     )
{
	const double  kappa_cast = *( const double* )kappa;
	const double* alpha1     = a;
	      double* pi1        = p;

	if ( inca == 1 && cdim_bcast == 1 )
	{
		bli_dpackm_armsme_int_2vlxk_body
		(
		  cdim, cdim_max, n, n_max,
		  kappa_cast, alpha1, lda, pi1, ldp
		);
	}
	else if ( lda == 1 && cdim_bcast == 1 )
	{
		bli_dpackm_armsme_int_2vlxk_trans
		(
		  cdim, cdim_max, n, n_max,
		  kappa_cast, alpha1, inca, pi1, ldp
		);
	}
	else
	{
		for ( dim_t j = 0; j < n; ++j )
		{
			for ( dim_t i = 0; i < cdim; ++i )
			for ( dim_t d = 0; d < cdim_bcast; ++d )
				pi1[ j*ldp + i*cdim_bcast + d ] = kappa_cast * alpha1[ j*lda + i*inca ];

			for ( dim_t i = cdim*cdim_bcast; i < cdim_max*cdim_bcast; ++i )
				pi1[ j*ldp + i ] = 0.0;
		}

		for ( dim_t j = n; j < n_max; ++j )
			for ( dim_t i = 0; i < cdim_max*cdim_bcast; ++i )
				pi1[ j*ldp + i ] = 0.0;
	}
}

// -- scomplex -----------------------------------------------------------------
//
// Complex packers produce the STANDARD interleaved format [r0 i0 r1 i1 ...] per
// k-slice (ldp in complex units, zero-padded), byte-compatible with the
// reference packm so that complex trsm/trmm/gemmtrsm (which share this ukr) keep
// working. conja negates the imaginary part; a possibly-complex kappa is applied
// as p = kappa * conj?(a). The transpose path treats each complex as one 64-bit
// unit so the (r,i) pair stays intact through the ZA64 transpose.

// Apply p = kappa * conj?(a) to one interleaved f32 vector x = [r0 i0 r1 i1 ...]
// (8 complex at SVL=512). vs = +1 (no conj) or -1 (conj), applied to the imag
// lanes. Uses fma (svmls/svmla) matching the reference's contracted scal2s; for
// a real kappa (vki == 0) this reduces to an exact scalar multiply.
#define CPACKM_SCALE_C32( x, pt, vkr, vki, vs ) \
	do { \
		svfloat32_t _re = svuzp1_f32( ( x ), ( x ) ); \
		svfloat32_t _im = svmul_f32_x( ( pt ), svuzp2_f32( ( x ), ( x ) ), ( vs ) ); \
		svfloat32_t _or = svmls_f32_x( ( pt ), svmul_f32_x( ( pt ), _re, ( vkr ) ), _im, ( vki ) ); \
		svfloat32_t _oi = svmla_f32_x( ( pt ), svmul_f32_x( ( pt ), _im, ( vkr ) ), _re, ( vki ) ); \
		( x ) = svzip1_f32( _or, _oi ); \
	} while ( 0 )

// Contiguous path (inca == 1): source column j is a contiguous interleaved
// complex run. Scale and store per 8-complex (VL-real) chunk.
__arm_locally_streaming
static void bli_cpackm_armsme_int_2vlxk_body
     (
       dim_t           cdim,
       dim_t           cdim_max,
       dim_t           n,
       dim_t           n_max,
       scomplex        kappa,
       conj_t          conja,
       const scomplex* a, inc_t lda,
       scomplex*       p, inc_t ldp
     )
{
	const uint64_t    vl      = svcntw();
	const svbool_t    pt      = svptrue_b32();
	const svfloat32_t vkr     = svdup_f32( kappa.real );
	const svfloat32_t vki     = svdup_f32( kappa.imag );
	const svfloat32_t vs      = svdup_f32( bli_is_conj( conja ) ? -1.0f : 1.0f );
	const svfloat32_t vzero   = svdup_f32( 0.0f );
	const dim_t       rdat    = 2 * cdim;
	const dim_t       rmax    = 2 * cdim_max;
	// Fast path for the common gemm pack (kappa == 1, no conjugation): copy only.
	const bool        noscale = !bli_is_conj( conja ) && kappa.real == 1.0f && kappa.imag == 0.0f;

	for ( dim_t j = 0; j < n; ++j )
	{
		const float* aj = ( const float* )( a + j*lda );
		float*       pj = ( float* )( p + j*ldp );

		for ( dim_t v = 0; v < rmax; v += ( dim_t )vl )
		{
			const svbool_t pld = svwhilelt_b32( ( uint64_t )v, ( uint64_t )rdat );
			const svbool_t pst = svwhilelt_b32( ( uint64_t )v, ( uint64_t )rmax );
			svfloat32_t x = svld1_f32( pld, aj + v );
			if ( !noscale ) CPACKM_SCALE_C32( x, pt, vkr, vki, vs );
			svst1_f32( pst, pj + v, x );
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		float* pj = ( float* )( p + j*ldp );
		for ( dim_t v = 0; v < rmax; v += ( dim_t )vl )
			svst1_f32( svwhilelt_b32( ( uint64_t )v, ( uint64_t )rmax ), pj + v, vzero );
	}
}

// Transpose path (lda == 1, inca != 1): ZA64 transpose keeping (r,i) intact.
__arm_new("za") __arm_locally_streaming
static void bli_cpackm_armsme_int_2vlxk_trans
     (
       dim_t           cdim,
       dim_t           cdim_max,
       dim_t           n,
       dim_t           n_max,
       scomplex        kappa,
       conj_t          conja,
       const scomplex* a, inc_t inca,
       scomplex*       p, inc_t ldp
     )
{
	const uint64_t    vl      = svcntd();   // complex (64-bit) elements per slice
	const svbool_t    pt      = svptrue_b32();
	const svfloat32_t vkr     = svdup_f32( kappa.real );
	const svfloat32_t vki     = svdup_f32( kappa.imag );
	const svfloat32_t vs      = svdup_f32( bli_is_conj( conja ) ? -1.0f : 1.0f );
	const svfloat64_t vzero64 = svdup_f64( 0.0 );
	const svfloat32_t vzero32 = svdup_f32( 0.0f );
	const uint64_t*   a64     = ( const uint64_t* )a;
	const bool        noscale = !bli_is_conj( conja ) && kappa.real == 1.0f && kappa.imag == 0.0f;

	for ( dim_t kb = 0; kb < n; kb += ( dim_t )vl )
	{
		const dim_t    kk = ( n - kb < ( dim_t )vl ) ? ( n - kb ) : ( dim_t )vl;
		const svbool_t pk = svwhilelt_b64( ( uint64_t )0, ( uint64_t )kk );

		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t sr = ( rb < cdim )
			                 ? ( ( cdim - rb < ( dim_t )vl ) ? ( cdim - rb ) : ( dim_t )vl )
			                 : 0;
			for ( dim_t s = 0; s < sr; ++s )
				svld1_hor_za64( 0, ( uint32_t )s, pk, ( const void* )( a64 + ( rb + s )*inca + kb ) );

			const svbool_t pr = svwhilelt_b64( ( uint64_t )rb, ( uint64_t )cdim );
			const dim_t    sw = ( cdim_max - rb < ( dim_t )vl ) ? ( cdim_max - rb ) : ( dim_t )vl;
			const svbool_t pst = svwhilelt_b32( ( uint64_t )0, ( uint64_t )( 2*sw ) );

			for ( dim_t q = 0; q < kk; ++q )
			{
				svfloat32_t x = svreinterpret_f32_f64(
				                   svread_ver_za64_f64_m( vzero64, pr, 0, ( uint32_t )q ) );
				if ( !noscale ) CPACKM_SCALE_C32( x, pt, vkr, vki, vs );
				svst1_f32( pst, ( float* )( p + ( kb + q )*ldp + rb ), x );
			}
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		float* pj = ( float* )( p + j*ldp );
		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t    sw  = ( cdim_max - rb < ( dim_t )vl ) ? ( cdim_max - rb ) : ( dim_t )vl;
			const svbool_t pst = svwhilelt_b32( ( uint64_t )0, ( uint64_t )( 2*sw ) );
			svst1_f32( pst, pj + 2*rb, vzero32 );
		}
	}
}

void bli_cpackm_armsme_int_2vlxk
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n,
             dim_t   n_max,
       const void*   kappa,
       const void*   a, inc_t inca, inc_t lda,
             void*   p,             inc_t ldp,
       const void*   params,
       const cntx_t* cntx
     )
{
	const scomplex  kappa_cast = *( const scomplex* )kappa;
	const scomplex* aa         = ( const scomplex* )a;
	      scomplex* pp         = ( scomplex* )p;

	if ( inca == 1 && cdim_bcast == 1 )
	{
		bli_cpackm_armsme_int_2vlxk_body
		( cdim, cdim_max, n, n_max, kappa_cast, conja, aa, lda, pp, ldp );
	}
	else if ( lda == 1 && cdim_bcast == 1 )
	{
		bli_cpackm_armsme_int_2vlxk_trans
		( cdim, cdim_max, n, n_max, kappa_cast, conja, aa, inca, pp, ldp );
	}
	else
	{
		const bool  cj = bli_is_conj( conja );
		const float kr = kappa_cast.real, ki = kappa_cast.imag;

		for ( dim_t j = 0; j < n; ++j )
		{
			float* pj = ( float* )( pp + j*ldp );
			for ( dim_t i = 0; i < cdim; ++i )
			{
				scomplex aij = aa[ j*lda + i*inca ];
				float ar = aij.real;
				float ai = cj ? -aij.imag : aij.imag;
				pj[ 2*i     ] = kr*ar - ki*ai;
				pj[ 2*i + 1 ] = kr*ai + ki*ar;
			}
			for ( dim_t i = cdim; i < cdim_max; ++i )
			{
				pj[ 2*i ] = 0.0f; pj[ 2*i + 1 ] = 0.0f;
			}
		}
		for ( dim_t j = n; j < n_max; ++j )
		{
			float* pj = ( float* )( pp + j*ldp );
			for ( dim_t i = 0; i < cdim_max; ++i )
			{
				pj[ 2*i ] = 0.0f; pj[ 2*i + 1 ] = 0.0f;
			}
		}
	}
}

// -- dcomplex -----------------------------------------------------------------
//
// Same design as scomplex, but each dcomplex is a 128-bit unit: the transpose
// uses ZA128 (validated on Apple clang 21) so the (r,i) f64 pair stays intact.
// NOTE: ZA128 slice ops are governed by a b64 predicate with TWO lanes per
// 128-bit element, so predicates count 2*(#dcomplex).

// Apply p = kappa * conj?(a) to one interleaved f64 vector x = [r0 i0 r1 i1 ...]
// (4 dcomplex at SVL=512). See CPACKM_SCALE_C32.
#define ZPACKM_SCALE_Z64( x, pt, vkr, vki, vs ) \
	do { \
		svfloat64_t _re = svuzp1_f64( ( x ), ( x ) ); \
		svfloat64_t _im = svmul_f64_x( ( pt ), svuzp2_f64( ( x ), ( x ) ), ( vs ) ); \
		svfloat64_t _or = svmls_f64_x( ( pt ), svmul_f64_x( ( pt ), _re, ( vkr ) ), _im, ( vki ) ); \
		svfloat64_t _oi = svmla_f64_x( ( pt ), svmul_f64_x( ( pt ), _im, ( vkr ) ), _re, ( vki ) ); \
		( x ) = svzip1_f64( _or, _oi ); \
	} while ( 0 )

__arm_locally_streaming
static void bli_zpackm_armsme_int_2vlxk_body
     (
       dim_t           cdim,
       dim_t           cdim_max,
       dim_t           n,
       dim_t           n_max,
       dcomplex        kappa,
       conj_t          conja,
       const dcomplex* a, inc_t lda,
       dcomplex*       p, inc_t ldp
     )
{
	const uint64_t    vl    = svcntd();
	const svbool_t    pt    = svptrue_b64();
	const svfloat64_t vkr   = svdup_f64( kappa.real );
	const svfloat64_t vki   = svdup_f64( kappa.imag );
	const svfloat64_t vs    = svdup_f64( bli_is_conj( conja ) ? -1.0 : 1.0 );
	const svfloat64_t vzero = svdup_f64( 0.0 );
	const dim_t       rdat  = 2 * cdim;
	const dim_t       rmax  = 2 * cdim_max;
	const bool        noscale = !bli_is_conj( conja ) && kappa.real == 1.0 && kappa.imag == 0.0;

	for ( dim_t j = 0; j < n; ++j )
	{
		const double* aj = ( const double* )( a + j*lda );
		double*       pj = ( double* )( p + j*ldp );

		for ( dim_t v = 0; v < rmax; v += ( dim_t )vl )
		{
			const svbool_t pld = svwhilelt_b64( ( uint64_t )v, ( uint64_t )rdat );
			const svbool_t pst = svwhilelt_b64( ( uint64_t )v, ( uint64_t )rmax );
			svfloat64_t x = svld1_f64( pld, aj + v );
			if ( !noscale ) ZPACKM_SCALE_Z64( x, pt, vkr, vki, vs );
			svst1_f64( pst, pj + v, x );
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		double* pj = ( double* )( p + j*ldp );
		for ( dim_t v = 0; v < rmax; v += ( dim_t )vl )
			svst1_f64( svwhilelt_b64( ( uint64_t )v, ( uint64_t )rmax ), pj + v, vzero );
	}
}

// Transpose path via ZA128 (128-bit dcomplex units; b64 predicates count 2 per).
__arm_new("za") __arm_locally_streaming
static void bli_zpackm_armsme_int_2vlxk_trans
     (
       dim_t           cdim,
       dim_t           cdim_max,
       dim_t           n,
       dim_t           n_max,
       dcomplex        kappa,
       conj_t          conja,
       const dcomplex* a, inc_t inca,
       dcomplex*       p, inc_t ldp
     )
{
	const uint64_t    vl    = svcntd() / 2;   // dcomplex (128-bit) elements per slice
	const svbool_t    pt    = svptrue_b64();
	const svfloat64_t vkr   = svdup_f64( kappa.real );
	const svfloat64_t vki   = svdup_f64( kappa.imag );
	const svfloat64_t vs    = svdup_f64( bli_is_conj( conja ) ? -1.0 : 1.0 );
	const svfloat64_t vzero = svdup_f64( 0.0 );
	const double*     ad    = ( const double* )a;
	const bool        noscale = !bli_is_conj( conja ) && kappa.real == 1.0 && kappa.imag == 0.0;

	for ( dim_t kb = 0; kb < n; kb += ( dim_t )vl )
	{
		const dim_t    kk = ( n - kb < ( dim_t )vl ) ? ( n - kb ) : ( dim_t )vl;
		const svbool_t pk = svwhilelt_b64( ( uint64_t )0, ( uint64_t )( 2*kk ) );

		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t ar = ( rb < cdim )
			                 ? ( ( cdim - rb < ( dim_t )vl ) ? ( cdim - rb ) : ( dim_t )vl )
			                 : 0;
			for ( dim_t s = 0; s < ar; ++s )
				svld1_hor_za128( 0, ( uint32_t )s, pk, ( const void* )( ad + ( rb + s )*2*inca + 2*kb ) );

			const svbool_t pr  = svwhilelt_b64( ( uint64_t )0, ( uint64_t )( 2*ar ) );
			const dim_t    sw  = ( cdim_max - rb < ( dim_t )vl ) ? ( cdim_max - rb ) : ( dim_t )vl;
			const svbool_t pst = svwhilelt_b64( ( uint64_t )0, ( uint64_t )( 2*sw ) );

			for ( dim_t q = 0; q < kk; ++q )
			{
				svfloat64_t x = svread_ver_za128_f64_m( vzero, pr, 0, ( uint32_t )q );
				if ( !noscale ) ZPACKM_SCALE_Z64( x, pt, vkr, vki, vs );
				svst1_f64( pst, ( double* )( p + ( kb + q )*ldp + rb ), x );
			}
		}
	}

	for ( dim_t j = n; j < n_max; ++j )
	{
		double* pj = ( double* )( p + j*ldp );
		for ( dim_t rb = 0; rb < cdim_max; rb += ( dim_t )vl )
		{
			const dim_t    sw  = ( cdim_max - rb < ( dim_t )vl ) ? ( cdim_max - rb ) : ( dim_t )vl;
			const svbool_t pst = svwhilelt_b64( ( uint64_t )0, ( uint64_t )( 2*sw ) );
			svst1_f64( pst, pj + 2*rb, vzero );
		}
	}
}

void bli_zpackm_armsme_int_2vlxk
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n,
             dim_t   n_max,
       const void*   kappa,
       const void*   a, inc_t inca, inc_t lda,
             void*   p,             inc_t ldp,
       const void*   params,
       const cntx_t* cntx
     )
{
	const dcomplex  kappa_cast = *( const dcomplex* )kappa;
	const dcomplex* aa         = ( const dcomplex* )a;
	      dcomplex* pp         = ( dcomplex* )p;

	if ( inca == 1 && cdim_bcast == 1 )
	{
		bli_zpackm_armsme_int_2vlxk_body
		( cdim, cdim_max, n, n_max, kappa_cast, conja, aa, lda, pp, ldp );
	}
	else if ( lda == 1 && cdim_bcast == 1 )
	{
		bli_zpackm_armsme_int_2vlxk_trans
		( cdim, cdim_max, n, n_max, kappa_cast, conja, aa, inca, pp, ldp );
	}
	else
	{
		const bool   cj = bli_is_conj( conja );
		const double kr = kappa_cast.real, ki = kappa_cast.imag;

		for ( dim_t j = 0; j < n; ++j )
		{
			double* pj = ( double* )( pp + j*ldp );
			for ( dim_t i = 0; i < cdim; ++i )
			{
				dcomplex aij = aa[ j*lda + i*inca ];
				double ar = aij.real;
				double ai = cj ? -aij.imag : aij.imag;
				pj[ 2*i     ] = kr*ar - ki*ai;
				pj[ 2*i + 1 ] = kr*ai + ki*ar;
			}
			for ( dim_t i = cdim; i < cdim_max; ++i )
			{
				pj[ 2*i ] = 0.0; pj[ 2*i + 1 ] = 0.0;
			}
		}
		for ( dim_t j = n; j < n_max; ++j )
		{
			double* pj = ( double* )( pp + j*ldp );
			for ( dim_t i = 0; i < cdim_max; ++i )
			{
				pj[ 2*i ] = 0.0; pj[ 2*i + 1 ] = 0.0;
			}
		}
	}
}
