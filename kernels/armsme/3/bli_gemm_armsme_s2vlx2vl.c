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

// SME2 outer-product sgemm microkernel, 2VL x 2VL (32x32 at SVL = 512).
//
// The 32x32 f32 accumulator lives in all four ZA32 tiles:
//
//              n: 0..VL-1   n: VL..2VL-1
//   m: 0..VL-1    za0          za2
//   m: VL..2VL-1  za1          za3
//
// Each k iteration loads one column of the packed A micro-panel (2 vectors)
// and one row of the packed B micro-panel (2 vectors) and issues four fmopa.
// m/n edge cases are handled with predicates; only rs_c != 1 goes through
// a temporary microtile.
//
// The streaming-mode switch and ZA allocation are per-call: this function
// is ABI-compatible with a plain C function (__arm_locally_streaming and
// __arm_new("za") do not change the calling convention), so it is safe to
// call through the generic gemm_ukr_ft pointer.

__arm_new("za") __arm_locally_streaming
static void bli_sgemm_armsme_2vlx2vl_body
     (
       dim_t        m,
       dim_t        n,
       dim_t        k,
       const float* alpha,
       const float* a,
       const float* b,
       const float* beta,
       float*       c,
       inc_t        cs_c
     )
{
	const uint64_t vl = svcntw();

	const svbool_t pm0 = svwhilelt_b32( ( uint64_t )0, ( uint64_t )m );
	const svbool_t pm1 = svwhilelt_b32( vl,            ( uint64_t )m );
	const svbool_t pn0 = svwhilelt_b32( ( uint64_t )0, ( uint64_t )n );
	const svbool_t pn1 = svwhilelt_b32( vl,            ( uint64_t )n );

	// The packed micro-panels are zero-padded to full MR/NR width, so the
	// k-loop can use unpredicated (full-width) multi-vector loads and fmopa;
	// the padding rows/columns accumulate zeros in ZA and are never stored.
	const svbool_t  pall = svptrue_b32();
	const svcount_t pc   = svptrue_c32();

	// ZA is already zero on entry: __arm_new("za") zero-initializes the tile
	// state per the Arm ACLE, so no explicit svzero_za() is needed here.
	dim_t l = 0;

	// Main loop: 2 k-iterations per pass, one 4-vector load per operand.
	for ( ; l + 1 < k; l += 2 )
	{
		svfloat32x4_t av = svld1_x4( pc, a );
		svfloat32x4_t bv = svld1_x4( pc, b );

		svmopa_za32_f32_m( 0, pall, pall, svget4( av, 0 ), svget4( bv, 0 ) );
		svmopa_za32_f32_m( 1, pall, pall, svget4( av, 1 ), svget4( bv, 0 ) );
		svmopa_za32_f32_m( 2, pall, pall, svget4( av, 0 ), svget4( bv, 1 ) );
		svmopa_za32_f32_m( 3, pall, pall, svget4( av, 1 ), svget4( bv, 1 ) );

		svmopa_za32_f32_m( 0, pall, pall, svget4( av, 2 ), svget4( bv, 2 ) );
		svmopa_za32_f32_m( 1, pall, pall, svget4( av, 3 ), svget4( bv, 2 ) );
		svmopa_za32_f32_m( 2, pall, pall, svget4( av, 2 ), svget4( bv, 3 ) );
		svmopa_za32_f32_m( 3, pall, pall, svget4( av, 3 ), svget4( bv, 3 ) );

		a += 4*vl;
		b += 4*vl;
	}

	for ( ; l < k; ++l )
	{
		svfloat32x2_t av = svld1_x2( pc, a );
		svfloat32x2_t bv = svld1_x2( pc, b );

		svmopa_za32_f32_m( 0, pall, pall, svget2( av, 0 ), svget2( bv, 0 ) );
		svmopa_za32_f32_m( 1, pall, pall, svget2( av, 1 ), svget2( bv, 0 ) );
		svmopa_za32_f32_m( 2, pall, pall, svget2( av, 0 ), svget2( bv, 1 ) );
		svmopa_za32_f32_m( 3, pall, pall, svget2( av, 1 ), svget2( bv, 1 ) );

		a += 2*vl;
		b += 2*vl;
	}

	const svfloat32_t valpha = svdup_f32( *alpha );
	const svfloat32_t vbeta  = svdup_f32( *beta );

	if ( *beta == 0.0f )
	{
		// beta == 0: overwrite C without reading it (it may be uninitialized).
		for ( dim_t j = 0; j < n; ++j )
		{
			float* cj = c + j*cs_c;

			svfloat32_t ab0, ab1;
			if ( ( uint64_t )j < vl )
			{
				ab0 = svread_ver_za32_f32_m( svundef_f32(), pm0, 0, ( uint32_t )j );
				ab1 = svread_ver_za32_f32_m( svundef_f32(), pm1, 1, ( uint32_t )j );
			}
			else
			{
				ab0 = svread_ver_za32_f32_m( svundef_f32(), pm0, 2, ( uint32_t )( j - vl ) );
				ab1 = svread_ver_za32_f32_m( svundef_f32(), pm1, 3, ( uint32_t )( j - vl ) );
			}

			svst1_f32( pm0, cj,      svmul_f32_x( pm0, ab0, valpha ) );
			svst1_f32( pm1, cj + vl, svmul_f32_x( pm1, ab1, valpha ) );
		}
	}
	else
	{
		for ( dim_t j = 0; j < n; ++j )
		{
			float* cj = c + j*cs_c;

			svfloat32_t ab0, ab1;
			if ( ( uint64_t )j < vl )
			{
				ab0 = svread_ver_za32_f32_m( svundef_f32(), pm0, 0, ( uint32_t )j );
				ab1 = svread_ver_za32_f32_m( svundef_f32(), pm1, 1, ( uint32_t )j );
			}
			else
			{
				ab0 = svread_ver_za32_f32_m( svundef_f32(), pm0, 2, ( uint32_t )( j - vl ) );
				ab1 = svread_ver_za32_f32_m( svundef_f32(), pm1, 3, ( uint32_t )( j - vl ) );
			}

			svfloat32_t c0 = svld1_f32( pm0, cj );
			svfloat32_t c1 = svld1_f32( pm1, cj + vl );

			c0 = svmla_f32_x( pm0, svmul_f32_x( pm0, ab0, valpha ), c0, vbeta );
			c1 = svmla_f32_x( pm1, svmul_f32_x( pm1, ab1, valpha ), c1, vbeta );

			svst1_f32( pm0, cj,      c0 );
			svst1_f32( pm1, cj + vl, c1 );
		}
	}
}

// Slow path: general row stride (rs_c != 1). The streaming body requires unit
// row stride, so route C through a temporary microtile. Kept in a separate
// noinline function so its 16 KiB _ct buffer (and the accompanying stack probe)
// never lands in the fast-path frame below.
__attribute__((noinline))
static void bli_sgemm_armsme_2vlx2vl_ct
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
	const dim_t nr = mr;

	GEMM_UKR_SETUP_CT_PRE( s, mr, nr, false, 1 );
	const bool _use_ct = true; // this helper is only reached when rs_c != 1
	GEMM_UKR_SETUP_CT_POST( s );

	bli_sgemm_armsme_2vlx2vl_body
	(
	  m, n, k,
	  ( const float* )alpha,
	  ( const float* )a,
	  ( const float* )b,
	  ( const float* )beta,
	  ( float* )c, cs_c
	);

	GEMM_UKR_FLUSH_CT( s );
}

void bli_sgemm_armsme_2vlx2vl
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
		bli_sgemm_armsme_2vlx2vl_body
		(
		  m, n, k,
		  ( const float* )alpha,
		  ( const float* )a,
		  ( const float* )b,
		  ( const float* )beta,
		  ( float* )c, cs_c
		);
		return;
	}

	bli_sgemm_armsme_2vlx2vl_ct( m, n, k, alpha, a, b, beta, c, rs_c, cs_c );
}
