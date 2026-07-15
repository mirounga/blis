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

// SME2 outer-product cgemm microkernel, 2VL x 1VL (32x16 at SVL = 512).
//
// A complex output microtile of MR x NR = 2VL x 1VL (complex elements) lives
// in all four ZA32 tiles, two per complex sub-tile (real + imaginary):
//
//                 n: 0..VL-1
//   m: 0..VL-1     za0 (re) / za1 (im)
//   m: VL..2VL-1   za2 (re) / za3 (im)
//
// The packed micro-panels use the standard interleaved complex format (the
// same one the reference packm and all other complex level-3 kernels consume),
// so this kernel de-interleaves each k-slice into planar real/imag vectors with
// svuzp (on the vector unit, overlapping the SME outer products). One svld1_x4
// loads a k-column of A (MR = 2VL complex) and one svld1_x2 loads a k-row of B
// (NR = VL complex). Each k iteration issues eight outer products (per sub-tile:
// two into the real accumulator via fmopa/fmops, two into the imaginary
// accumulator via fmopa), forming
//   C_re += A_re*B_re - A_im*B_im,  C_im += A_re*B_im + A_im*B_re.
//
// Conjugation of A/B is applied at pack time, so this kernel is conj-agnostic.
// Complex alpha and beta are applied in the epilogue. C is stored interleaved
// (r,i), so the epilogue de-interleaves the read of C with svuzp and
// re-interleaves the store with svzip (svld2/svst2 are illegal in streaming
// mode). Only rs_c != 1 goes through a temporary microtile.

__arm_new("za") __arm_locally_streaming
static void bli_cgemm_armsme_2vlx1vl_body
     (
       dim_t           m,
       dim_t           n,
       dim_t           k,
       const scomplex* alpha,
       const float*    a,
       const float*    b,
       const scomplex* beta,
       scomplex*       c,
       inc_t           cs_c
     )
{
	const uint64_t vl = svcntw();

	// The packed micro-panels are zero-padded to full MR/NR width, so the
	// k-loop uses unpredicated (full-width) multi-vector loads and outer
	// products; the padding rows/columns accumulate zeros in ZA.
	const svbool_t  pall = svptrue_b32();
	const svcount_t pc   = svptrue_c32();

	// ZA is already zero on entry (__arm_new("za") zero-initializes it).
	for ( dim_t l = 0; l < k; ++l )
	{
		// A interleaved k-slice: MR = 2VL complex = [ r i r i ... ] = 4 vectors.
		svfloat32x4_t av = svld1_x4( pc, a );
		// B interleaved k-slice: NR = VL complex = [ r i r i ... ] = 2 vectors.
		svfloat32x2_t bv = svld1_x2( pc, b );

		// De-interleave into planar real/imag halves (uzp1 = reals, uzp2 =
		// imags). av0/av1 hold m-block 0 (rows 0..VL-1), av2/av3 m-block 1.
		const svfloat32_t a_re0 = svuzp1_f32( svget4( av, 0 ), svget4( av, 1 ) );
		const svfloat32_t a_im0 = svuzp2_f32( svget4( av, 0 ), svget4( av, 1 ) );
		const svfloat32_t a_re1 = svuzp1_f32( svget4( av, 2 ), svget4( av, 3 ) );
		const svfloat32_t a_im1 = svuzp2_f32( svget4( av, 2 ), svget4( av, 3 ) );
		const svfloat32_t b_re  = svuzp1_f32( svget2( bv, 0 ), svget2( bv, 1 ) );
		const svfloat32_t b_im  = svuzp2_f32( svget2( bv, 0 ), svget2( bv, 1 ) );

		// m-block 0: za0 = re, za1 = im.
		svmopa_za32_f32_m( 0, pall, pall, a_re0, b_re );
		svmops_za32_f32_m( 0, pall, pall, a_im0, b_im );
		svmopa_za32_f32_m( 1, pall, pall, a_re0, b_im );
		svmopa_za32_f32_m( 1, pall, pall, a_im0, b_re );

		// m-block 1: za2 = re, za3 = im.
		svmopa_za32_f32_m( 2, pall, pall, a_re1, b_re );
		svmops_za32_f32_m( 2, pall, pall, a_im1, b_im );
		svmopa_za32_f32_m( 3, pall, pall, a_re1, b_im );
		svmopa_za32_f32_m( 3, pall, pall, a_im1, b_re );

		a += 4*vl;
		b += 2*vl;
	}

	const svfloat32_t valpha_re = svdup_f32( alpha->real );
	const svfloat32_t valpha_im = svdup_f32( alpha->imag );
	const svfloat32_t vbeta_re  = svdup_f32( beta->real );
	const svfloat32_t vbeta_im  = svdup_f32( beta->imag );
	const bool        beta0     = ( beta->real == 0.0f && beta->imag == 0.0f );

	// SME tile indices must be compile-time constants, so the two m-blocks are
	// unrolled via a macro parameterized by literal (re,im) tile indices.
	#define CGEMM_EPI_MB( re_tile, im_tile, mb ) \
	do { \
		svfloat32_t ab_re = svread_ver_za32_f32_m( svundef_f32(), pall, (re_tile), ( uint32_t )j ); \
		svfloat32_t ab_im = svread_ver_za32_f32_m( svundef_f32(), pall, (im_tile), ( uint32_t )j ); \
		/* out = alpha * ab  (complex). */ \
		svfloat32_t out_re = svmls_f32_x( pall, svmul_f32_x( pall, ab_re, valpha_re ), ab_im, valpha_im ); \
		svfloat32_t out_im = svmla_f32_x( pall, svmul_f32_x( pall, ab_im, valpha_re ), ab_re, valpha_im ); \
		const dim_t   md  = m - ( dim_t )( (mb)*vl ); \
		const dim_t   mm  = md <= 0 ? 0 : ( md < ( dim_t )vl ? md : ( dim_t )vl ); \
		float*        cfb = cf + (mb)*2*vl; \
		const svbool_t plo = svwhilelt_b32( ( uint64_t )0, ( uint64_t )( 2*mm ) ); \
		const svbool_t phi = svwhilelt_b32( vl,            ( uint64_t )( 2*mm ) ); \
		if ( !beta0 ) \
		{ \
			/* out += beta * C  (complex); C is interleaved (r,i). */ \
			svfloat32_t l0   = svld1_f32( plo, cfb ); \
			svfloat32_t l1   = svld1_f32( phi, cfb + vl ); \
			svfloat32_t c_re = svuzp1_f32( l0, l1 ); \
			svfloat32_t c_im = svuzp2_f32( l0, l1 ); \
			out_re = svmla_f32_x( pall, out_re, c_re, vbeta_re ); \
			out_re = svmls_f32_x( pall, out_re, c_im, vbeta_im ); \
			out_im = svmla_f32_x( pall, out_im, c_re, vbeta_im ); \
			out_im = svmla_f32_x( pall, out_im, c_im, vbeta_re ); \
		} \
		svfloat32_t v0 = svzip1_f32( out_re, out_im ); \
		svfloat32_t v1 = svzip2_f32( out_re, out_im ); \
		svst1_f32( plo, cfb,      v0 ); \
		svst1_f32( phi, cfb + vl, v1 ); \
	} while ( 0 )

	for ( dim_t j = 0; j < n; ++j )
	{
		float* cf = ( float* )( c + j*cs_c );

		CGEMM_EPI_MB( 0, 1, 0 );  // m-block 0: za0 = re, za1 = im
		CGEMM_EPI_MB( 2, 3, 1 );  // m-block 1: za2 = re, za3 = im
	}

	#undef CGEMM_EPI_MB
}

// Slow path: general row stride (rs_c != 1). The streaming body requires unit
// row stride, so route C through a temporary microtile. Kept in a separate
// noinline function so its _ct buffer never lands in the fast-path frame below.
__attribute__((noinline))
static void bli_cgemm_armsme_2vlx1vl_ct
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
	const dim_t mr = 2 * ( dim_t )svcntsw();
	const dim_t nr =     ( dim_t )svcntsw();

	GEMM_UKR_SETUP_CT_PRE( c, mr, nr, false, 1 );
	const bool _use_ct = true; // this helper is only reached when rs_c != 1
	GEMM_UKR_SETUP_CT_POST( c );

	bli_cgemm_armsme_2vlx1vl_body
	(
	  m, n, k,
	  ( const scomplex* )alpha,
	  ( const float* )a,
	  ( const float* )b,
	  ( const scomplex* )beta,
	  ( scomplex* )c, cs_c
	);

	GEMM_UKR_FLUSH_CT( c );
}

void bli_cgemm_armsme_2vlx1vl
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
	// Fast path: unit row stride (column-major C). The streaming body handles
	// m/n edges with predicates and any cs_c, so call it directly with no
	// temporary microtile, keeping this frame tiny.
	if ( rs_c == 1 )
	{
		bli_cgemm_armsme_2vlx1vl_body
		(
		  m, n, k,
		  ( const scomplex* )alpha,
		  ( const float* )a,
		  ( const float* )b,
		  ( const scomplex* )beta,
		  ( scomplex* )c, cs_c
		);
		return;
	}

	bli_cgemm_armsme_2vlx1vl_ct( m, n, k, alpha, a, b, beta, c, rs_c, cs_c );
}
