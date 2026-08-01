#ifndef BLS_SC_2C_ARMV8A_H
#define BLS_SC_2C_ARMV8A_H

#include <stdint.h>

/*
 * Extra output metadata for the two-destination Strassen microkernels.
 *
 * The first output continues to use the ordinary GEMM ukernel arguments.
 * The second output is passed through auxinfo_t::params, which keeps these
 * functions ABI-compatible with gemm_ukr_ft. The caller folds the sign of
 * the first destination into alpha and supplies the sign of the second
 * destination relative to the first in relative_sign.
 *
 * Both output tiles must be full, row-stored microtiles.
 */
typedef struct
{
	void*       c2;
	const void* beta2;
	inc_t       rs_c2;
	int32_t     relative_sign;
} bls_sc_2c_params_t;

bool bls_sc_2c_armv8a_is_compatible
     (
       num_t         dt,
       gemm_ukr_ft   gemm_ukr,
       const cntx_t* cntx
     );

void bls_sc_sgemm_armv8a_asm_12x8r_2c
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
     );

void bls_sc_dgemm_armv8a_asm_8x6r_2c
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
     );

#endif
