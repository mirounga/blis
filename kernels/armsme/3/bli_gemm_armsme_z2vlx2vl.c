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

// NOTE: Apple clang 21 does not define __ARM_FEATURE_SME_F64F64 even when
// -march=...+sme-f64f64 is in effect (the codegen works; only the ACLE
// macro is missing), so __ARM_FEATURE_SME2 is accepted as an alternative.
// The armsme kernel set is always compiled with +sme-f64f64 (see
// config/applesme/make_defs.mk), and the context init registers this
// kernel only when FEAT_SME_F64F64 is detected at runtime.
#if defined(__ARM_FEATURE_SME_F64F64) || defined(__ARM_FEATURE_SME2)

#include <arm_sme.h>

// SME2 outer-product zgemm microkernel, 2VL x 2VL (16x16 at SVL = 512).
// Requires FEAT_SME_F64F64 (f64 fmopa/fmops into ZA64 tiles).
//
// A complex output microtile of MR x NR = 2VL x 2VL (complex elements) lives
// in all eight ZA64 tiles, two per complex sub-tile (real + imaginary),
// arranged as a 2x2 grid of VL x VL complex sub-tiles. With sub-tile index
// s = mb + 2*nb (m-block mb in {0,1}, n-block nb in {0,1}):
//
//                 n: 0..VL-1              n: VL..2VL-1
//   m: 0..VL-1     za0 (re) / za1 (im)    za4 (re) / za5 (im)
//   m: VL..2VL-1   za2 (re) / za3 (im)    za6 (re) / za7 (im)
//
// The packed micro-panels use the standard interleaved complex format (shared
// with the reference packm and all other complex level-3 kernels), so this
// kernel de-interleaves each k-slice into planar real/imag vectors with svuzp
// (on the vector unit, overlapping the SME outer products). One svld1_x4 loads
// a k-column of A (MR = 2VL complex) and one svld1_x4 loads a k-row of B
// (NR = 2VL complex). Each k iteration issues sixteen outer products, forming
// per sub-tile
//   C_re += A_re*B_re - A_im*B_im,  C_im += A_re*B_im + A_im*B_re.
//
// Conjugation is applied at pack time (this kernel is conj-agnostic). Complex
// alpha and beta are applied in the epilogue; C is stored interleaved (r,i),
// so the epilogue de-interleaves the C read with svuzp and re-interleaves the
// store with svzip. Only rs_c != 1 goes through a temporary microtile.

__arm_new("za") __arm_locally_streaming
static void bli_zgemm_armsme_2vlx2vl_body
     (
       dim_t           m,
       dim_t           n,
       dim_t           k,
       const dcomplex* alpha,
       const double*   a,
       const double*   b,
       const dcomplex* beta,
       dcomplex*       c,
       inc_t           cs_c
     )
{
	const uint64_t vl = svcntd();

	const svbool_t  pall = svptrue_b64();
	const svcount_t pc   = svptrue_c64();

	// ZA is already zero on entry (__arm_new("za") zero-initializes it).
	for ( dim_t l = 0; l < k; ++l )
	{
		// A interleaved k-slice: MR = 2VL complex = [ r i r i ... ].
		svfloat64x4_t av = svld1_x4( pc, a );
		// B interleaved k-slice: NR = 2VL complex = [ r i r i ... ].
		svfloat64x4_t bv = svld1_x4( pc, b );

		// De-interleave into planar real/imag halves. av0/av1 hold m-block 0,
		// av2/av3 m-block 1; bv0/bv1 hold n-block 0, bv2/bv3 n-block 1.
		const svfloat64_t a_re0 = svuzp1_f64( svget4( av, 0 ), svget4( av, 1 ) );
		const svfloat64_t a_im0 = svuzp2_f64( svget4( av, 0 ), svget4( av, 1 ) );
		const svfloat64_t a_re1 = svuzp1_f64( svget4( av, 2 ), svget4( av, 3 ) );
		const svfloat64_t a_im1 = svuzp2_f64( svget4( av, 2 ), svget4( av, 3 ) );
		const svfloat64_t b_re0 = svuzp1_f64( svget4( bv, 0 ), svget4( bv, 1 ) );
		const svfloat64_t b_im0 = svuzp2_f64( svget4( bv, 0 ), svget4( bv, 1 ) );
		const svfloat64_t b_re1 = svuzp1_f64( svget4( bv, 2 ), svget4( bv, 3 ) );
		const svfloat64_t b_im1 = svuzp2_f64( svget4( bv, 2 ), svget4( bv, 3 ) );

		// SME tile indices must be compile-time constants, so all four complex
		// sub-tiles (eight ZA64 tiles) are unrolled with literal indices.
		// (m0,n0) -> za0/za1
		svmopa_za64_f64_m( 0, pall, pall, a_re0, b_re0 );
		svmops_za64_f64_m( 0, pall, pall, a_im0, b_im0 );
		svmopa_za64_f64_m( 1, pall, pall, a_re0, b_im0 );
		svmopa_za64_f64_m( 1, pall, pall, a_im0, b_re0 );
		// (m1,n0) -> za2/za3
		svmopa_za64_f64_m( 2, pall, pall, a_re1, b_re0 );
		svmops_za64_f64_m( 2, pall, pall, a_im1, b_im0 );
		svmopa_za64_f64_m( 3, pall, pall, a_re1, b_im0 );
		svmopa_za64_f64_m( 3, pall, pall, a_im1, b_re0 );
		// (m0,n1) -> za4/za5
		svmopa_za64_f64_m( 4, pall, pall, a_re0, b_re1 );
		svmops_za64_f64_m( 4, pall, pall, a_im0, b_im1 );
		svmopa_za64_f64_m( 5, pall, pall, a_re0, b_im1 );
		svmopa_za64_f64_m( 5, pall, pall, a_im0, b_re1 );
		// (m1,n1) -> za6/za7
		svmopa_za64_f64_m( 6, pall, pall, a_re1, b_re1 );
		svmops_za64_f64_m( 6, pall, pall, a_im1, b_im1 );
		svmopa_za64_f64_m( 7, pall, pall, a_re1, b_im1 );
		svmopa_za64_f64_m( 7, pall, pall, a_im1, b_re1 );

		a += 4*vl;
		b += 4*vl;
	}

	const svfloat64_t valpha_re = svdup_f64( alpha->real );
	const svfloat64_t valpha_im = svdup_f64( alpha->imag );
	const svfloat64_t vbeta_re  = svdup_f64( beta->real );
	const svfloat64_t vbeta_im  = svdup_f64( beta->imag );
	const bool        beta0     = ( beta->real == 0.0 && beta->imag == 0.0 );

	// SME tile indices must be compile-time constants; the per-m-block epilogue
	// is a macro taking literal (re,im) tile indices, dispatched by n-block cb.
	#define ZGEMM_EPI_MB( re_tile, im_tile, mb ) \
	do { \
		svfloat64_t ab_re = svread_ver_za64_f64_m( svundef_f64(), pall, (re_tile), jj ); \
		svfloat64_t ab_im = svread_ver_za64_f64_m( svundef_f64(), pall, (im_tile), jj ); \
		/* out = alpha * ab  (complex). */ \
		svfloat64_t out_re = svmls_f64_x( pall, svmul_f64_x( pall, ab_re, valpha_re ), ab_im, valpha_im ); \
		svfloat64_t out_im = svmla_f64_x( pall, svmul_f64_x( pall, ab_im, valpha_re ), ab_re, valpha_im ); \
		const dim_t   md  = m - ( dim_t )( (mb)*vl ); \
		const dim_t   mm  = md <= 0 ? 0 : ( md < ( dim_t )vl ? md : ( dim_t )vl ); \
		double*       cfb = cf + (mb)*2*vl; \
		const svbool_t plo = svwhilelt_b64( ( uint64_t )0, ( uint64_t )( 2*mm ) ); \
		const svbool_t phi = svwhilelt_b64( vl,            ( uint64_t )( 2*mm ) ); \
		if ( !beta0 ) \
		{ \
			/* out += beta * C  (complex); C is interleaved (r,i). */ \
			svfloat64_t l0   = svld1_f64( plo, cfb ); \
			svfloat64_t l1   = svld1_f64( phi, cfb + vl ); \
			svfloat64_t c_re = svuzp1_f64( l0, l1 ); \
			svfloat64_t c_im = svuzp2_f64( l0, l1 ); \
			out_re = svmla_f64_x( pall, out_re, c_re, vbeta_re ); \
			out_re = svmls_f64_x( pall, out_re, c_im, vbeta_im ); \
			out_im = svmla_f64_x( pall, out_im, c_re, vbeta_im ); \
			out_im = svmla_f64_x( pall, out_im, c_im, vbeta_re ); \
		} \
		svfloat64_t v0 = svzip1_f64( out_re, out_im ); \
		svfloat64_t v1 = svzip2_f64( out_re, out_im ); \
		svst1_f64( plo, cfb,      v0 ); \
		svst1_f64( phi, cfb + vl, v1 ); \
	} while ( 0 )

	for ( dim_t j = 0; j < n; ++j )
	{
		double* cf = ( double* )( c + j*cs_c );

		const uint64_t cb = ( uint64_t )j / vl;         // n-block (0 or 1)
		const uint32_t jj = ( uint32_t )( ( uint64_t )j - cb*vl );

		if ( cb == 0 )
		{
			ZGEMM_EPI_MB( 0, 1, 0 );  // (m0,n0) -> za0/za1
			ZGEMM_EPI_MB( 2, 3, 1 );  // (m1,n0) -> za2/za3
		}
		else
		{
			ZGEMM_EPI_MB( 4, 5, 0 );  // (m0,n1) -> za4/za5
			ZGEMM_EPI_MB( 6, 7, 1 );  // (m1,n1) -> za6/za7
		}
	}

	#undef ZGEMM_EPI_MB
}

// Slow path: general row stride (rs_c != 1). The streaming body requires unit
// row stride, so route C through a temporary microtile.
__attribute__((noinline))
static void bli_zgemm_armsme_2vlx2vl_ct
     (
             dim_t   m,
             dim_t   n,
             dim_t   k,
       const void*   alpha,
       const void*   a,
       const void*   b,
       const void*   beta,
             void*   c, inc_t rs_c, inc_t cs_c
     )
{
	const dim_t mr = 2 * ( dim_t )svcntsd();
	const dim_t nr = 2 * ( dim_t )svcntsd();

	GEMM_UKR_SETUP_CT_PRE( z, mr, nr, false, 1 );
	const bool _use_ct = true; // this helper is only reached when rs_c != 1
	GEMM_UKR_SETUP_CT_POST( z );

	bli_zgemm_armsme_2vlx2vl_body
	(
	  m, n, k,
	  ( const dcomplex* )alpha,
	  ( const double* )a,
	  ( const double* )b,
	  ( const dcomplex* )beta,
	  ( dcomplex* )c, cs_c
	);

	GEMM_UKR_FLUSH_CT( z );
}

void bli_zgemm_armsme_2vlx2vl
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	if ( rs_c == 1 )
	{
		bli_zgemm_armsme_2vlx2vl_body
		(
		  m, n, k,
		  ( const dcomplex* )alpha,
		  ( const double* )a,
		  ( const double* )b,
		  ( const dcomplex* )beta,
		  ( dcomplex* )c, cs_c
		);
		return;
	}

	bli_zgemm_armsme_2vlx2vl_ct( m, n, k, alpha, a, b, beta, c, rs_c, cs_c );
}

#endif // __ARM_FEATURE_SME_F64F64 || __ARM_FEATURE_SME2
