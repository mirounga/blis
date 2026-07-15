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

void bli_cntx_init_applesme( cntx_t* cntx )
{
	// Set default kernel blocksizes and functions.
	bli_cntx_init_applesme_ref( cntx );

	// The SME2 kernels require the Scalable Matrix Extension. If it is not
	// present (e.g. when this subconfig is forced on unsupported hardware,
	// or when autodetecting within the arm64 family), keep the reference
	// kernels registered above.
	uint32_t family, model, features = 0;
	bli_cpuid_query( &family, &model, &features );

	if ( ! bli_cpuid_has_features( features, FEATURE_SME2 ) )
		return;

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// -------------------------------------------------------------------------

	// f64 outer products additionally require FEAT_SME_F64F64 (present on
	// Apple M4 Pro/Max? and all M5), otherwise double stays on reference
	// kernels.
	const bool has_f64f64 = bli_cpuid_has_features( features, FEATURE_SME_F64F64 );

	// Register blocksizes derived from the streaming vector length
	// (SVL = 512 bits -> svlw = 16, svld = 8 -> MR x NR = 32x32 for float,
	// 16x32 for double).
	const dim_t svlw  = bli_armsme_svlw();
	const dim_t svld  = bli_armsme_svld();
	const dim_t m_r_s = 2 * svlw;
	const dim_t n_r_s = 2 * svlw;
	const dim_t m_r_d = has_f64f64 ? 2 * svld : -1;
	const dim_t n_r_d = has_f64f64 ? 4 * svld : -1;
	// Complex register blocksizes (complex elements): cgemm 2VL x 1VL, zgemm
	// 2VL x 2VL. Each complex sub-tile uses two ZA tiles (real + imaginary),
	// so cgemm fills the 4 ZA32 tiles (2 sub-tiles) and zgemm the 8 ZA64 tiles
	// (4 sub-tiles). zgemm requires FEAT_SME_F64F64.
	const dim_t m_r_c = 2 * svlw;
	const dim_t n_r_c =     svlw;
	const dim_t m_r_z = has_f64f64 ? 2 * svld : -1;
	const dim_t n_r_z = has_f64f64 ? 2 * svld : -1;

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_armsme_2vlx2vl,
	  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_armsme_2vlx1vl,

	  // level-1m
	  BLIS_PACKM_KER, BLIS_FLOAT,    bli_spackm_armsme_int_2vlxk,
	  BLIS_PACKM_KER, BLIS_DOUBLE,   bli_dpackm_armsme_int_2vlxk,
	  BLIS_PACKM_KER, BLIS_SCOMPLEX, bli_cpackm_armsme_int_2vlxk,

	  BLIS_VA_END
	);

	if ( has_f64f64 )
	{
		bli_cntx_set_ukrs
		(
		  cntx,
		  BLIS_GEMM_UKR,  BLIS_DOUBLE,   bli_dgemm_armsme_2vlx4vl,
		  BLIS_GEMM_UKR,  BLIS_DCOMPLEX, bli_zgemm_armsme_2vlx2vl,
		  BLIS_PACKM_KER, BLIS_DCOMPLEX, bli_zpackm_armsme_int_2vlxk,
		  BLIS_VA_END
		);
	}

	// The complex packers produce the STANDARD interleaved packed format (the
	// same one the reference packm and complex trsm/trmm/gemmtrsm consume), so
	// registering them is safe; they simply vectorize the contiguous and
	// transpose cases the reference packed scalar-ish.

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,    FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE,   FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_SCOMPLEX, FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	// (-1 keeps the reference value for that datatype.)
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ], m_r_s, m_r_d, m_r_c, m_r_z );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ], n_r_s, n_r_d, n_r_c, n_r_z );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],  512, has_f64f64 ?  512 : -1,  256, has_f64f64 ?  256 : -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  2048, has_f64f64 ? 2048 : -1, 1024, has_f64f64 ? 1024 : -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4096, has_f64f64 ? 2048 : -1, 4096, has_f64f64 ? 2048 : -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  cntx,

	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

	  BLIS_VA_END
	);
}
