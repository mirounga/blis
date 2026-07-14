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
