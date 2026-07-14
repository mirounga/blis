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

// SME2 outer-product dgemm microkernel, 2VL x 4VL (16x32 at SVL = 512).
// Requires FEAT_SME_F64F64 (f64 fmopa into ZA64 tiles).
//
// The 16x32 f64 accumulator lives in all eight ZA64 tiles:
//
//             n: 0..VL-1  VL..2VL-1  2VL..3VL-1  3VL..4VL-1
//   m: 0..VL-1    za0        za2        za4         za6
//   m: VL..2VL-1  za1        za3        za5         za7
//
// Each k iteration loads one column of the packed A micro-panel (2 vectors)
// and one row of the packed B micro-panel (4 vectors) and issues eight
// fmopa. m/n edge cases are handled with predicates; only rs_c != 1 goes
// through a temporary microtile.

__arm_new("za") __arm_locally_streaming
static void bli_dgemm_armsme_2vlx4vl_body
     (
       dim_t         m,
       dim_t         n,
       dim_t         k,
       const double* alpha,
       const double* a,
       const double* b,
       const double* beta,
       double*       c,
       inc_t         cs_c
     )
{
	const uint64_t vl = svcntd();

	const svbool_t pm0 = svwhilelt_b64( ( uint64_t )0, ( uint64_t )m );
	const svbool_t pm1 = svwhilelt_b64( vl,            ( uint64_t )m );
	const svbool_t pn0 = svwhilelt_b64( ( uint64_t )0, ( uint64_t )n );
	const svbool_t pn1 = svwhilelt_b64( vl,            ( uint64_t )n );
	const svbool_t pn2 = svwhilelt_b64( 2*vl,          ( uint64_t )n );
	const svbool_t pn3 = svwhilelt_b64( 3*vl,          ( uint64_t )n );

	// The packed micro-panels are zero-padded to full MR/NR width, so the
	// k-loop can use unpredicated (full-width) multi-vector loads and fmopa;
	// the padding rows/columns accumulate zeros in ZA and are never stored.
	const svbool_t  pall = svptrue_b64();
	const svcount_t pc   = svptrue_c64();

	svzero_za();

	dim_t l = 0;

	// Main loop: 2 k-iterations per pass; A is one 4-vector load, B is two.
	for ( ; l + 1 < k; l += 2 )
	{
		svfloat64x4_t av  = svld1_x4( pc, a );
		svfloat64x4_t bv0 = svld1_x4( pc, b );
		svfloat64x4_t bv1 = svld1_x4( pc, b + 4*vl );

		svmopa_za64_f64_m( 0, pall, pall, svget4( av, 0 ), svget4( bv0, 0 ) );
		svmopa_za64_f64_m( 1, pall, pall, svget4( av, 1 ), svget4( bv0, 0 ) );
		svmopa_za64_f64_m( 2, pall, pall, svget4( av, 0 ), svget4( bv0, 1 ) );
		svmopa_za64_f64_m( 3, pall, pall, svget4( av, 1 ), svget4( bv0, 1 ) );
		svmopa_za64_f64_m( 4, pall, pall, svget4( av, 0 ), svget4( bv0, 2 ) );
		svmopa_za64_f64_m( 5, pall, pall, svget4( av, 1 ), svget4( bv0, 2 ) );
		svmopa_za64_f64_m( 6, pall, pall, svget4( av, 0 ), svget4( bv0, 3 ) );
		svmopa_za64_f64_m( 7, pall, pall, svget4( av, 1 ), svget4( bv0, 3 ) );

		svmopa_za64_f64_m( 0, pall, pall, svget4( av, 2 ), svget4( bv1, 0 ) );
		svmopa_za64_f64_m( 1, pall, pall, svget4( av, 3 ), svget4( bv1, 0 ) );
		svmopa_za64_f64_m( 2, pall, pall, svget4( av, 2 ), svget4( bv1, 1 ) );
		svmopa_za64_f64_m( 3, pall, pall, svget4( av, 3 ), svget4( bv1, 1 ) );
		svmopa_za64_f64_m( 4, pall, pall, svget4( av, 2 ), svget4( bv1, 2 ) );
		svmopa_za64_f64_m( 5, pall, pall, svget4( av, 3 ), svget4( bv1, 2 ) );
		svmopa_za64_f64_m( 6, pall, pall, svget4( av, 2 ), svget4( bv1, 3 ) );
		svmopa_za64_f64_m( 7, pall, pall, svget4( av, 3 ), svget4( bv1, 3 ) );

		a += 4*vl;
		b += 8*vl;
	}

	for ( ; l < k; ++l )
	{
		svfloat64x2_t av = svld1_x2( pc, a );
		svfloat64x4_t bv = svld1_x4( pc, b );

		svmopa_za64_f64_m( 0, pall, pall, svget2( av, 0 ), svget4( bv, 0 ) );
		svmopa_za64_f64_m( 1, pall, pall, svget2( av, 1 ), svget4( bv, 0 ) );
		svmopa_za64_f64_m( 2, pall, pall, svget2( av, 0 ), svget4( bv, 1 ) );
		svmopa_za64_f64_m( 3, pall, pall, svget2( av, 1 ), svget4( bv, 1 ) );
		svmopa_za64_f64_m( 4, pall, pall, svget2( av, 0 ), svget4( bv, 2 ) );
		svmopa_za64_f64_m( 5, pall, pall, svget2( av, 1 ), svget4( bv, 2 ) );
		svmopa_za64_f64_m( 6, pall, pall, svget2( av, 0 ), svget4( bv, 3 ) );
		svmopa_za64_f64_m( 7, pall, pall, svget2( av, 1 ), svget4( bv, 3 ) );

		a += 2*vl;
		b += 4*vl;
	}

	const svfloat64_t valpha = svdup_f64( *alpha );
	const svfloat64_t vbeta  = svdup_f64( *beta );
	const bool        beta0  = ( *beta == 0.0 );

	for ( dim_t j = 0; j < n; ++j )
	{
		double* cj = c + j*cs_c;

		const uint64_t cb = ( uint64_t )j / vl;
		const uint32_t jj = ( uint32_t )( ( uint64_t )j - cb*vl );

		svfloat64_t ab0, ab1;
		switch ( cb )
		{
			case 0:
				ab0 = svread_ver_za64_f64_m( svundef_f64(), pm0, 0, jj );
				ab1 = svread_ver_za64_f64_m( svundef_f64(), pm1, 1, jj );
				break;
			case 1:
				ab0 = svread_ver_za64_f64_m( svundef_f64(), pm0, 2, jj );
				ab1 = svread_ver_za64_f64_m( svundef_f64(), pm1, 3, jj );
				break;
			case 2:
				ab0 = svread_ver_za64_f64_m( svundef_f64(), pm0, 4, jj );
				ab1 = svread_ver_za64_f64_m( svundef_f64(), pm1, 5, jj );
				break;
			default:
				ab0 = svread_ver_za64_f64_m( svundef_f64(), pm0, 6, jj );
				ab1 = svread_ver_za64_f64_m( svundef_f64(), pm1, 7, jj );
				break;
		}

		ab0 = svmul_f64_x( pm0, ab0, valpha );
		ab1 = svmul_f64_x( pm1, ab1, valpha );

		if ( !beta0 )
		{
			// beta != 0: accumulate into the existing C values.
			svfloat64_t c0 = svld1_f64( pm0, cj );
			svfloat64_t c1 = svld1_f64( pm1, cj + vl );

			ab0 = svmla_f64_x( pm0, ab0, c0, vbeta );
			ab1 = svmla_f64_x( pm1, ab1, c1, vbeta );
		}

		svst1_f64( pm0, cj,      ab0 );
		svst1_f64( pm1, cj + vl, ab1 );
	}
}

void bli_dgemm_armsme_2vlx4vl
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
	const dim_t mr = 2 * ( dim_t )svcntsd();
	const dim_t nr = 2 * mr;

	// The streaming body handles m/n edges with predicates and any cs_c,
	// but requires unit row stride; use a temporary microtile only when
	// rs_c != 1.
	GEMM_UKR_SETUP_CT_PRE( d, mr, nr, false, 1 );
	const bool _use_ct = ( rs_c != 1 );
	GEMM_UKR_SETUP_CT_POST( d );

	bli_dgemm_armsme_2vlx4vl_body
	(
	  m, n, k,
	  ( const double* )alpha,
	  ( const double* )a,
	  ( const double* )b,
	  ( const double* )beta,
	  ( double* )c, cs_c
	);

	GEMM_UKR_FLUSH_CT( d );
}

#endif // __ARM_FEATURE_SME_F64F64 || __ARM_FEATURE_SME2
