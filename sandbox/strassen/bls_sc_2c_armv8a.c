/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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
#include "bls_sc_2c_armv8a.h"

#include <assert.h>
#include <stdint.h>

bool bls_sc_2c_armv8a_is_compatible
     (
       num_t         dt,
       gemm_ukr_ft   gemm_ukr,
       const cntx_t* cntx
     )
{
#if defined(__aarch64__) && defined(BLIS_KERNELS_ARMV8A)
	const dim_t mr = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t nr = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const bool  rp =
	    bli_cntx_get_ukr_prefs_dt( dt, BLIS_GEMM_UKR_ROW_PREF, cntx );

	if ( !rp ) return FALSE;

	if ( dt == BLIS_FLOAT )
		return mr == 12 && nr == 8 &&
		       gemm_ukr == ( gemm_ukr_ft )bli_sgemm_armv8a_asm_12x8r;

	if ( dt == BLIS_DOUBLE )
		return mr == 8 && nr == 6 &&
		       gemm_ukr == ( gemm_ukr_ft )bli_dgemm_armv8a_asm_8x6r;
#else
	( void )dt;
	( void )gemm_ukr;
	( void )cntx;
#endif

	return FALSE;
}

#if defined(__aarch64__) && defined(BLIS_KERNELS_ARMV8A)

#include "../../kernels/armv8a/3/armv8a_asm_utils.h"
#include "../../kernels/armv8a/3/armv8a_asm_d2x2.h"

/*
 * These compute macros are kept instruction-for-instruction identical to
 * bli_{s,d}gemm_armv8a_asm_{12x8r,8x6r}. Only the C epilogue differs.
 */
#define BLS_SGEMM_12X8_MKER_LOOP(C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,C60,C61,C70,C71,C80,C81,C90,C91,CA0,CA1,CB0,CB1,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  SGEMM_4X4_NANOKERNEL(C00,C10,C20,C30,B0,A0) \
  SGEMM_4X4_NANOKERNEL(C01,C11,C21,C31,B1,A0) \
  DGEMM_LOAD1V_ ##LOADNEXT (A0,AADDR,ASHIFT) \
  SGEMM_4X4_NANOKERNEL(C40,C50,C60,C70,B0,A1) \
  SGEMM_4X4_NANOKERNEL(C41,C51,C61,C71,B1,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (A1,AADDR,ASHIFT+16) \
  SGEMM_4X4_NANOKERNEL(C80,C90,CA0,CB0,B0,A2) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  SGEMM_4X4_NANOKERNEL(C81,C91,CA1,CB1,B1,A2)

#define BLS_DGEMM_8X6_MKER_LOOP(C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  DGEMM_2X2_NANOKERNEL(C00,C10,B0,A0) \
  DGEMM_2X2_NANOKERNEL(C20,C30,B0,A1) \
  DGEMM_2X2_NANOKERNEL(C01,C11,B1,A0) \
  DGEMM_2X2_NANOKERNEL(C21,C31,B1,A1) \
  DGEMM_2X2_NANOKERNEL(C02,C12,B2,A0) \
  DGEMM_2X2_NANOKERNEL(C22,C32,B2,A1) \
  DGEMM_LOAD2V_ ##LOADNEXT (A0,A1,AADDR,ASHIFT) \
  DGEMM_2X2_NANOKERNEL(C40,C50,B0,A2) \
  DGEMM_2X2_NANOKERNEL(C60,C70,B0,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL(C41,C51,B1,A2) \
  DGEMM_2X2_NANOKERNEL(C61,C71,B1,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B1,BADDR,BSHIFT+16) \
  DGEMM_2X2_NANOKERNEL(C42,C52,B2,A2) \
  DGEMM_2X2_NANOKERNEL(C62,C72,B2,A3)

#define BLS_DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define BLS_DGEMM_LOAD1V_load(V1,ADDR,IMM) DLOAD1V(V1,ADDR,IMM)
#define BLS_DGEMM_LOAD2V_noload(V1,V2,ADDR,IMM)
#define BLS_DGEMM_LOAD2V_load(V1,V2,ADDR,IMM) \
  BLS_DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  BLS_DGEMM_LOAD1V_load(V2,ADDR,IMM+16)

/*
 * The nanokernel macros concatenate DGEMM_LOAD{1,2}V_ with LOADNEXT.
 * Redirect those names to the local copies used by this translation unit.
 */
#undef DGEMM_LOAD1V_noload
#undef DGEMM_LOAD1V_load
#undef DGEMM_LOAD2V_noload
#undef DGEMM_LOAD2V_load
#define DGEMM_LOAD1V_noload BLS_DGEMM_LOAD1V_noload
#define DGEMM_LOAD1V_load   BLS_DGEMM_LOAD1V_load
#define DGEMM_LOAD2V_noload BLS_DGEMM_LOAD2V_noload
#define DGEMM_LOAD2V_load   BLS_DGEMM_LOAD2V_load

#define BLS_SLOADC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define BLS_SSTOREC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

#define BLS_DLOADC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
  DLOAD1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define BLS_DSTOREC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
  DSTORE1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

#define BLS_PRFMC_FWD(CADDR,RSC,LASTB) \
" prfm PLDL1KEEP, ["#CADDR"]           \n\t" \
" prfm PLDL1KEEP, ["#CADDR", "#LASTB"] \n\t" \
" add  "#CADDR", "#CADDR", "#RSC"      \n\t"

#define BLS_SADD2V(D0,D1,S0,S1) \
" fadd v"#D0".4s, v"#D0".4s, v"#S0".4s \n\t" \
" fadd v"#D1".4s, v"#D1".4s, v"#S1".4s \n\t"
#define BLS_SSUB2V(D0,D1,S0,S1) \
" fsub v"#D0".4s, v"#D0".4s, v"#S0".4s \n\t" \
" fsub v"#D1".4s, v"#D1".4s, v"#S1".4s \n\t"
#define BLS_SNEG2V(D0,D1,S0,S1) \
" fneg v"#D0".4s, v"#S0".4s \n\t" \
" fneg v"#D1".4s, v"#S1".4s \n\t"

#define BLS_DADD3V(D0,D1,D2,S0,S1,S2) \
" fadd v"#D0".2d, v"#D0".2d, v"#S0".2d \n\t" \
" fadd v"#D1".2d, v"#D1".2d, v"#S1".2d \n\t" \
" fadd v"#D2".2d, v"#D2".2d, v"#S2".2d \n\t"
#define BLS_DSUB3V(D0,D1,D2,S0,S1,S2) \
" fsub v"#D0".2d, v"#D0".2d, v"#S0".2d \n\t" \
" fsub v"#D1".2d, v"#D1".2d, v"#S1".2d \n\t" \
" fsub v"#D2".2d, v"#D2".2d, v"#S2".2d \n\t"
#define BLS_DNEG3V(D0,D1,D2,S0,S1,S2) \
" fneg v"#D0".2d, v"#S0".2d \n\t" \
" fneg v"#D1".2d, v"#S1".2d \n\t" \
" fneg v"#D2".2d, v"#S2".2d \n\t"

/*
 * Three SGEMM rows (six vectors) or two DGEMM rows (six vectors) fit in
 * v26-v31. Accumulators v0-v23 therefore remain intact for output two.
 */
#define BLS_SUPDATE3(OP,A0,A1,A2,A3,A4,A5,LOAD,STORE,RSC) \
  BLS_SLOADC_2V_R_FWD(26,27,LOAD,0,RSC) \
  BLS_SLOADC_2V_R_FWD(28,29,LOAD,0,RSC) \
  BLS_SLOADC_2V_R_FWD(30,31,LOAD,0,RSC) \
  SSCALE2V(26,27,25,0) \
  SSCALE2V(28,29,25,0) \
  SSCALE2V(30,31,25,0) \
  OP(26,27,A0,A1) \
  OP(28,29,A2,A3) \
  OP(30,31,A4,A5) \
  BLS_SSTOREC_2V_R_FWD(26,27,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(28,29,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(30,31,STORE,0,RSC)

#define BLS_SUPDATE3_UNIT(OP,A0,A1,A2,A3,A4,A5,LOAD,STORE,RSC) \
  BLS_SLOADC_2V_R_FWD(26,27,LOAD,0,RSC) \
  BLS_SLOADC_2V_R_FWD(28,29,LOAD,0,RSC) \
  BLS_SLOADC_2V_R_FWD(30,31,LOAD,0,RSC) \
  OP(26,27,A0,A1) \
  OP(28,29,A2,A3) \
  OP(30,31,A4,A5) \
  BLS_SSTOREC_2V_R_FWD(26,27,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(28,29,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(30,31,STORE,0,RSC)

#define BLS_SSTORE3(A0,A1,A2,A3,A4,A5,STORE,RSC) \
  BLS_SSTOREC_2V_R_FWD(A0,A1,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(A2,A3,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(A4,A5,STORE,0,RSC)

#define BLS_SNEGSTORE3(A0,A1,A2,A3,A4,A5,STORE,RSC) \
  BLS_SNEG2V(26,27,A0,A1) \
  BLS_SNEG2V(28,29,A2,A3) \
  BLS_SNEG2V(30,31,A4,A5) \
  BLS_SSTOREC_2V_R_FWD(26,27,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(28,29,STORE,0,RSC) \
  BLS_SSTOREC_2V_R_FWD(30,31,STORE,0,RSC)

#define BLS_DUPDATE2(OP,A0,A1,A2,A3,A4,A5,LOAD,STORE,RSC) \
  BLS_DLOADC_3V_R_FWD(26,27,28,LOAD,0,RSC) \
  BLS_DLOADC_3V_R_FWD(29,30,31,LOAD,0,RSC) \
  DSCALE2V(26,27,25,0) \
  DSCALE2V(28,29,25,0) \
  DSCALE2V(30,31,25,0) \
  OP(26,27,28,A0,A1,A2) \
  OP(29,30,31,A3,A4,A5) \
  BLS_DSTOREC_3V_R_FWD(26,27,28,STORE,0,RSC) \
  BLS_DSTOREC_3V_R_FWD(29,30,31,STORE,0,RSC)

#define BLS_DUPDATE2_UNIT(OP,A0,A1,A2,A3,A4,A5,LOAD,STORE,RSC) \
  BLS_DLOADC_3V_R_FWD(26,27,28,LOAD,0,RSC) \
  BLS_DLOADC_3V_R_FWD(29,30,31,LOAD,0,RSC) \
  OP(26,27,28,A0,A1,A2) \
  OP(29,30,31,A3,A4,A5) \
  BLS_DSTOREC_3V_R_FWD(26,27,28,STORE,0,RSC) \
  BLS_DSTOREC_3V_R_FWD(29,30,31,STORE,0,RSC)

#define BLS_DSTORE2(A0,A1,A2,A3,A4,A5,STORE,RSC) \
  BLS_DSTOREC_3V_R_FWD(A0,A1,A2,STORE,0,RSC) \
  BLS_DSTOREC_3V_R_FWD(A3,A4,A5,STORE,0,RSC)

#define BLS_DNEGSTORE2(A0,A1,A2,A3,A4,A5,STORE,RSC) \
  BLS_DNEG3V(26,27,28,A0,A1,A2) \
  BLS_DNEG3V(29,30,31,A3,A4,A5) \
  BLS_DSTOREC_3V_R_FWD(26,27,28,STORE,0,RSC) \
  BLS_DSTOREC_3V_R_FWD(29,30,31,STORE,0,RSC)

void bls_sc_sgemm_armv8a_asm_12x8r_2c
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const bls_sc_2c_params_t* params =
	    ( const bls_sc_2c_params_t* )bli_auxinfo_params( data );

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );
	const void* beta2  = params->beta2;
	      void* c2     = params->c2;

	uint64_t k_mker      = k / 4;
	uint64_t k_left      = k % 4;
	uint64_t rs_c        = rs_c0;
	uint64_t rs_c2       = params->rs_c2;
	uint64_t c2_negative = params->relative_sign < 0;

	( void )m;
	( void )n;
	( void )cs_c0;
	( void )cntx;

	__asm__ volatile
	(
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #12                         \n\t"
" mov             x3, #8                          \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t"
" ldr             x10, %[c2]                      \n\t"
" ldr             x7, %[rs_c2]                    \n\t"
" lsl             x2, x2, #2                      \n\t"
" lsl             x3, x3, #2                      \n\t"
" lsl             x6, x6, #2                      \n\t"
" lsl             x7, x7, #2                      \n\t"
" mov             x9, x5                          \n\t"
" mov             x11, x10                        \n\t"
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"

#define BLS_SGEMM_12X8_MKER_LOOP_LOC(A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  BLS_SGEMM_12X8_MKER_LOOP(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT)

LABEL(BLS_SLOAD_ABC)
" cmp             x4, #0                          \n\t"
BEQ(BLS_SCLEAR_CCOLS)
" ldr             q24, [x0, #16*0]                \n\t"
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q27, [x0, #16*0]                \n\t"
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
BLS_PRFMC_FWD(x9,x6,32)
BLS_PRFMC_FWD(x11,x7,32)
" cmp             x4, #0                          \n\t"
" ldr             q28, [x1, #16*0]                \n\t"
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q30, [x1, #16*0]                \n\t"
" ldr             q31, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
LABEL(BLS_SCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
BEQ(BLS_SK_LEFT_LOOP)

#define BLS_SGEMM_12X8_MKER_LOOP_LOC_FWD(A0,A1,A2,B0,B1) \
  BLS_SGEMM_12X8_MKER_LOOP_LOC(A0,A1,A2,B0,B1,x0,16,x1,0,load) \
 "add             x0, x0, x2                      \n\t" \
 "ldr             q"#A2", [x0, #16*0]             \n\t" \
 "ldr             q"#B1", [x1, #16*1]             \n\t" \
 "add             x1, x1, x3                      \n\t"

LABEL(BLS_SK_MKER_LOOP)
BLS_SGEMM_12X8_MKER_LOOP_LOC_FWD(24,25,26,28,29)
BLS_SGEMM_12X8_MKER_LOOP_LOC_FWD(27,24,25,30,31)
" subs            x4, x4, #1                      \n\t"
BEQ(BLS_SFIN_MKER_LOOP)
BLS_SGEMM_12X8_MKER_LOOP_LOC_FWD(26,27,24,28,29)
BLS_SGEMM_12X8_MKER_LOOP_LOC_FWD(25,26,27,30,31)
BRANCH(BLS_SK_MKER_LOOP)

LABEL(BLS_SFIN_MKER_LOOP)
BLS_SGEMM_12X8_MKER_LOOP_LOC(26,27,24,28,29,xzr,-1,xzr,-1,noload)
" ldr             q26, [x0, #16*1]                \n\t"
" ldr             q27, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
BLS_SGEMM_12X8_MKER_LOOP_LOC(25,26,27,30,31,xzr,-1,xzr,-1,noload)

LABEL(BLS_SK_LEFT_LOOP)
" cmp             x8, #0                          \n\t"
BEQ(BLS_SWRITE_MEM_PREP)
" ldr             q24, [x0, #16*0]                \n\t"
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t"
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
BLS_SGEMM_12X8_MKER_LOOP_LOC(24,25,26,28,29,xzr,-1,xzr,-1,noload)
BRANCH(BLS_SK_LEFT_LOOP)

LABEL(BLS_SWRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t"
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v24.4s}, [x4]                  \n\t"
" ld1r            {v25.4s}, [x8]                  \n\t"
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL1STRM, [x0, 64*0]           \n\t"
" prfm            PLDL1STRM, [x0, 64*1]           \n\t"
" prfm            PLDL1STRM, [x0, 64*2]           \n\t"
" prfm            PLDL1STRM, [x1, 64*0]           \n\t"
" prfm            PLDL1STRM, [x1, 64*1]           \n\t"
" prfm            PLDL1STRM, [x1, 64*3]           \n\t"
" fmov            d26, #1.0                       \n\t"
" fcvt            s26, d26                        \n\t"
" fcmp            s24, s26                        \n\t"
BEQ(BLS_SUNIT_ALPHA)
SSCALE8V(0,1,2,3,4,5,6,7,24,0)
SSCALE8V(8,9,10,11,12,13,14,15,24,0)
SSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(BLS_SUNIT_ALPHA)
" fmov            s24, #1.0                       \n\t"

/* C0 = beta0*C0 + product. Preserve v0-v23. */
" mov             x9, x5                          \n\t"
" fcmp            s25, #0.0                       \n\t"
BEQ(BLS_SC0_ZERO_BETA)
" fcmp            s25, s24                        \n\t"
BEQ(BLS_SC0_UNIT_BETA)
BLS_SUPDATE3(BLS_SADD2V,0,1,2,3,4,5,x9,x5,x6)
BLS_SUPDATE3(BLS_SADD2V,6,7,8,9,10,11,x9,x5,x6)
BLS_SUPDATE3(BLS_SADD2V,12,13,14,15,16,17,x9,x5,x6)
BLS_SUPDATE3(BLS_SADD2V,18,19,20,21,22,23,x9,x5,x6)
BRANCH(BLS_SC0_DONE)
LABEL(BLS_SC0_UNIT_BETA)
BLS_SUPDATE3_UNIT(BLS_SADD2V,0,1,2,3,4,5,x9,x5,x6)
BLS_SUPDATE3_UNIT(BLS_SADD2V,6,7,8,9,10,11,x9,x5,x6)
BLS_SUPDATE3_UNIT(BLS_SADD2V,12,13,14,15,16,17,x9,x5,x6)
BLS_SUPDATE3_UNIT(BLS_SADD2V,18,19,20,21,22,23,x9,x5,x6)
BRANCH(BLS_SC0_DONE)
LABEL(BLS_SC0_ZERO_BETA)
BLS_SSTORE3(0,1,2,3,4,5,x5,x6)
BLS_SSTORE3(6,7,8,9,10,11,x5,x6)
BLS_SSTORE3(12,13,14,15,16,17,x5,x6)
BLS_SSTORE3(18,19,20,21,22,23,x5,x6)
LABEL(BLS_SC0_DONE)

/* Select the signed C2 epilogue once, outside all element updates. */
" ldr             x4, %[c2_negative]               \n\t"
" cmp             x4, #0                          \n\t"
BNE(BLS_SC2_NEGATIVE)

/* C2 = beta2*C2 + product. */
" ldr             x8, %[beta2]                    \n\t"
" ld1r            {v25.4s}, [x8]                  \n\t"
" mov             x11, x10                        \n\t"
" fcmp            s25, #0.0                       \n\t"
BEQ(BLS_SC2_POS_ZERO_BETA)
" fcmp            s25, s24                        \n\t"
BEQ(BLS_SC2_POS_UNIT_BETA)
BLS_SUPDATE3(BLS_SADD2V,0,1,2,3,4,5,x11,x10,x7)
BLS_SUPDATE3(BLS_SADD2V,6,7,8,9,10,11,x11,x10,x7)
BLS_SUPDATE3(BLS_SADD2V,12,13,14,15,16,17,x11,x10,x7)
BLS_SUPDATE3(BLS_SADD2V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_SEND_WRITE_MEM)
LABEL(BLS_SC2_POS_UNIT_BETA)
BLS_SUPDATE3_UNIT(BLS_SADD2V,0,1,2,3,4,5,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SADD2V,6,7,8,9,10,11,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SADD2V,12,13,14,15,16,17,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SADD2V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_SEND_WRITE_MEM)
LABEL(BLS_SC2_POS_ZERO_BETA)
BLS_SSTORE3(0,1,2,3,4,5,x10,x7)
BLS_SSTORE3(6,7,8,9,10,11,x10,x7)
BLS_SSTORE3(12,13,14,15,16,17,x10,x7)
BLS_SSTORE3(18,19,20,21,22,23,x10,x7)
BRANCH(BLS_SEND_WRITE_MEM)

/* C2 = beta2*C2 - product. */
LABEL(BLS_SC2_NEGATIVE)
" ldr             x8, %[beta2]                    \n\t"
" ld1r            {v25.4s}, [x8]                  \n\t"
" mov             x11, x10                        \n\t"
" fcmp            s25, #0.0                       \n\t"
BEQ(BLS_SC2_NEG_ZERO_BETA)
" fcmp            s25, s24                        \n\t"
BEQ(BLS_SC2_NEG_UNIT_BETA)
BLS_SUPDATE3(BLS_SSUB2V,0,1,2,3,4,5,x11,x10,x7)
BLS_SUPDATE3(BLS_SSUB2V,6,7,8,9,10,11,x11,x10,x7)
BLS_SUPDATE3(BLS_SSUB2V,12,13,14,15,16,17,x11,x10,x7)
BLS_SUPDATE3(BLS_SSUB2V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_SEND_WRITE_MEM)
LABEL(BLS_SC2_NEG_UNIT_BETA)
BLS_SUPDATE3_UNIT(BLS_SSUB2V,0,1,2,3,4,5,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SSUB2V,6,7,8,9,10,11,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SSUB2V,12,13,14,15,16,17,x11,x10,x7)
BLS_SUPDATE3_UNIT(BLS_SSUB2V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_SEND_WRITE_MEM)
LABEL(BLS_SC2_NEG_ZERO_BETA)
BLS_SNEGSTORE3(0,1,2,3,4,5,x10,x7)
BLS_SNEGSTORE3(6,7,8,9,10,11,x10,x7)
BLS_SNEGSTORE3(12,13,14,15,16,17,x10,x7)
BLS_SNEGSTORE3(18,19,20,21,22,23,x10,x7)

LABEL(BLS_SEND_WRITE_MEM)
	:
	: [a]           "m" (a),
	  [b]           "m" (b),
	  [c]           "m" (c),
	  [c2]          "m" (c2),
	  [rs_c]        "m" (rs_c),
	  [rs_c2]       "m" (rs_c2),
	  [k_mker]      "m" (k_mker),
	  [k_left]      "m" (k_left),
	  [alpha]       "m" (alpha),
	  [beta]        "m" (beta),
	  [beta2]       "m" (beta2),
	  [c2_negative] "m" (c2_negative),
	  [a_next]      "m" (a_next),
	  [b_next]      "m" (b_next)
	: "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11",
	  "v0","v1","v2","v3","v4","v5","v6","v7",
	  "v8","v9","v10","v11","v12","v13","v14","v15",
	  "v16","v17","v18","v19","v20","v21","v22","v23",
	  "v24","v25","v26","v27","v28","v29","v30","v31",
	  "cc","memory"
	);
}

void bls_sc_dgemm_armv8a_asm_8x6r_2c
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const bls_sc_2c_params_t* params =
	    ( const bls_sc_2c_params_t* )bli_auxinfo_params( data );

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );
	const void* beta2  = params->beta2;
	      void* c2     = params->c2;

	uint64_t k_mker      = k / 4;
	uint64_t k_left      = k % 4;
	uint64_t rs_c        = rs_c0;
	uint64_t rs_c2       = params->rs_c2;
	uint64_t c2_negative = params->relative_sign < 0;

	( void )m;
	( void )n;
	( void )cs_c0;
	( void )cntx;

	__asm__ volatile
	(
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #8                          \n\t"
" mov             x3, #6                          \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t"
" ldr             x10, %[c2]                      \n\t"
" ldr             x7, %[rs_c2]                    \n\t"
" lsl             x2, x2, #3                      \n\t"
" lsl             x3, x3, #3                      \n\t"
" lsl             x6, x6, #3                      \n\t"
" lsl             x7, x7, #3                      \n\t"
" mov             x9, x5                          \n\t"
" mov             x11, x10                        \n\t"
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
BLS_PRFMC_FWD(x9,x6,40)
BLS_PRFMC_FWD(x11,x7,40)
" ldr             x4, %[k_mker]                   \n\t"
" ldr             x8, %[k_left]                   \n\t"

#define BLS_DGEMM_8X6_MKER_LOOP_LOC(A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT) \
  BLS_DGEMM_8X6_MKER_LOOP(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT)

LABEL(BLS_DLOAD_ABC)
" cmp             x4, #0                          \n\t"
BEQ(BLS_DCLEAR_CCOLS)
" ldr             q24, [x0, #16*0]                \n\t"
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t"
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q31, [x1, #16*0]                \n\t"
LABEL(BLS_DCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
BEQ(BLS_DK_LEFT_LOOP)

#define BLS_DGEMM_8X6_MKER_LOOP_LOC_FWD(A0,A1,A2,A3,B0,B1,B2) \
  BLS_DGEMM_8X6_MKER_LOOP_LOC(A0,A1,A2,A3,B0,B1,B2,x0,0,x1,16,load) \
 "add             x1, x1, x3                      \n\t" \
 "ldr             q"#B2", [x1, #16*0]             \n\t" \
 "ldr             q"#A2", [x0, #16*2]             \n\t" \
 "ldr             q"#A3", [x0, #16*3]             \n\t" \
 "add             x0, x0, x2                      \n\t"

LABEL(BLS_DK_MKER_LOOP)
BLS_DGEMM_8X6_MKER_LOOP_LOC_FWD(24,25,26,27,28,29,30)
BLS_DGEMM_8X6_MKER_LOOP_LOC_FWD(24,25,26,27,31,28,29)
" subs            x4, x4, #1                      \n\t"
BEQ(BLS_DFIN_MKER_LOOP)
BLS_DGEMM_8X6_MKER_LOOP_LOC_FWD(24,25,26,27,30,31,28)
BLS_DGEMM_8X6_MKER_LOOP_LOC_FWD(24,25,26,27,29,30,31)
BRANCH(BLS_DK_MKER_LOOP)

LABEL(BLS_DFIN_MKER_LOOP)
BLS_DGEMM_8X6_MKER_LOOP_LOC(24,25,26,27,30,31,28,x0,0,x1,16,load)
" add             x1, x1, x3                      \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
BLS_DGEMM_8X6_MKER_LOOP_LOC(24,25,26,27,29,30,31,xzr,-1,xzr,-1,noload)

LABEL(BLS_DK_LEFT_LOOP)
" cmp             x8, #0                          \n\t"
BEQ(BLS_DWRITE_MEM_PREP)
" ldr             q24, [x0, #16*0]                \n\t"
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" ldr             q27, [x0, #16*3]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t"
" ldr             q29, [x1, #16*1]                \n\t"
" ldr             q30, [x1, #16*2]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
BLS_DGEMM_8X6_MKER_LOOP_LOC(24,25,26,27,28,29,30,xzr,-1,xzr,-1,noload)
BRANCH(BLS_DK_LEFT_LOOP)

LABEL(BLS_DWRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t"
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v24.2d}, [x4]                  \n\t"
" ld1r            {v25.2d}, [x8]                  \n\t"
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL1STRM, [x0, 64*0]           \n\t"
" prfm            PLDL1STRM, [x0, 64*1]           \n\t"
" prfm            PLDL1STRM, [x0, 64*2]           \n\t"
" prfm            PLDL1STRM, [x1, 64*0]           \n\t"
" prfm            PLDL1STRM, [x1, 64*1]           \n\t"
" prfm            PLDL1STRM, [x1, 64*3]           \n\t"
" fmov            d26, #1.0                       \n\t"
" fcmp            d24, d26                        \n\t"
BEQ(BLS_DUNIT_ALPHA)
DSCALE8V(0,1,2,3,4,5,6,7,24,0)
DSCALE8V(8,9,10,11,12,13,14,15,24,0)
DSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(BLS_DUNIT_ALPHA)
" fmov            d24, #1.0                       \n\t"

/* C0 = beta0*C0 + product. Preserve v0-v23. */
" mov             x9, x5                          \n\t"
" fcmp            d25, #0.0                       \n\t"
BEQ(BLS_DC0_ZERO_BETA)
" fcmp            d25, d24                        \n\t"
BEQ(BLS_DC0_UNIT_BETA)
BLS_DUPDATE2(BLS_DADD3V,0,1,2,3,4,5,x9,x5,x6)
BLS_DUPDATE2(BLS_DADD3V,6,7,8,9,10,11,x9,x5,x6)
BLS_DUPDATE2(BLS_DADD3V,12,13,14,15,16,17,x9,x5,x6)
BLS_DUPDATE2(BLS_DADD3V,18,19,20,21,22,23,x9,x5,x6)
BRANCH(BLS_DC0_DONE)
LABEL(BLS_DC0_UNIT_BETA)
BLS_DUPDATE2_UNIT(BLS_DADD3V,0,1,2,3,4,5,x9,x5,x6)
BLS_DUPDATE2_UNIT(BLS_DADD3V,6,7,8,9,10,11,x9,x5,x6)
BLS_DUPDATE2_UNIT(BLS_DADD3V,12,13,14,15,16,17,x9,x5,x6)
BLS_DUPDATE2_UNIT(BLS_DADD3V,18,19,20,21,22,23,x9,x5,x6)
BRANCH(BLS_DC0_DONE)
LABEL(BLS_DC0_ZERO_BETA)
BLS_DSTORE2(0,1,2,3,4,5,x5,x6)
BLS_DSTORE2(6,7,8,9,10,11,x5,x6)
BLS_DSTORE2(12,13,14,15,16,17,x5,x6)
BLS_DSTORE2(18,19,20,21,22,23,x5,x6)
LABEL(BLS_DC0_DONE)

" ldr             x4, %[c2_negative]               \n\t"
" cmp             x4, #0                          \n\t"
BNE(BLS_DC2_NEGATIVE)

/* C2 = beta2*C2 + product. */
" ldr             x8, %[beta2]                    \n\t"
" ld1r            {v25.2d}, [x8]                  \n\t"
" mov             x11, x10                        \n\t"
" fcmp            d25, #0.0                       \n\t"
BEQ(BLS_DC2_POS_ZERO_BETA)
" fcmp            d25, d24                        \n\t"
BEQ(BLS_DC2_POS_UNIT_BETA)
BLS_DUPDATE2(BLS_DADD3V,0,1,2,3,4,5,x11,x10,x7)
BLS_DUPDATE2(BLS_DADD3V,6,7,8,9,10,11,x11,x10,x7)
BLS_DUPDATE2(BLS_DADD3V,12,13,14,15,16,17,x11,x10,x7)
BLS_DUPDATE2(BLS_DADD3V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_DEND_WRITE_MEM)
LABEL(BLS_DC2_POS_UNIT_BETA)
BLS_DUPDATE2_UNIT(BLS_DADD3V,0,1,2,3,4,5,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DADD3V,6,7,8,9,10,11,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DADD3V,12,13,14,15,16,17,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DADD3V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_DEND_WRITE_MEM)
LABEL(BLS_DC2_POS_ZERO_BETA)
BLS_DSTORE2(0,1,2,3,4,5,x10,x7)
BLS_DSTORE2(6,7,8,9,10,11,x10,x7)
BLS_DSTORE2(12,13,14,15,16,17,x10,x7)
BLS_DSTORE2(18,19,20,21,22,23,x10,x7)
BRANCH(BLS_DEND_WRITE_MEM)

/* C2 = beta2*C2 - product. */
LABEL(BLS_DC2_NEGATIVE)
" ldr             x8, %[beta2]                    \n\t"
" ld1r            {v25.2d}, [x8]                  \n\t"
" mov             x11, x10                        \n\t"
" fcmp            d25, #0.0                       \n\t"
BEQ(BLS_DC2_NEG_ZERO_BETA)
" fcmp            d25, d24                        \n\t"
BEQ(BLS_DC2_NEG_UNIT_BETA)
BLS_DUPDATE2(BLS_DSUB3V,0,1,2,3,4,5,x11,x10,x7)
BLS_DUPDATE2(BLS_DSUB3V,6,7,8,9,10,11,x11,x10,x7)
BLS_DUPDATE2(BLS_DSUB3V,12,13,14,15,16,17,x11,x10,x7)
BLS_DUPDATE2(BLS_DSUB3V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_DEND_WRITE_MEM)
LABEL(BLS_DC2_NEG_UNIT_BETA)
BLS_DUPDATE2_UNIT(BLS_DSUB3V,0,1,2,3,4,5,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DSUB3V,6,7,8,9,10,11,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DSUB3V,12,13,14,15,16,17,x11,x10,x7)
BLS_DUPDATE2_UNIT(BLS_DSUB3V,18,19,20,21,22,23,x11,x10,x7)
BRANCH(BLS_DEND_WRITE_MEM)
LABEL(BLS_DC2_NEG_ZERO_BETA)
BLS_DNEGSTORE2(0,1,2,3,4,5,x10,x7)
BLS_DNEGSTORE2(6,7,8,9,10,11,x10,x7)
BLS_DNEGSTORE2(12,13,14,15,16,17,x10,x7)
BLS_DNEGSTORE2(18,19,20,21,22,23,x10,x7)

LABEL(BLS_DEND_WRITE_MEM)
	:
	: [a]           "m" (a),
	  [b]           "m" (b),
	  [c]           "m" (c),
	  [c2]          "m" (c2),
	  [rs_c]        "m" (rs_c),
	  [rs_c2]       "m" (rs_c2),
	  [k_mker]      "m" (k_mker),
	  [k_left]      "m" (k_left),
	  [alpha]       "m" (alpha),
	  [beta]        "m" (beta),
	  [beta2]       "m" (beta2),
	  [c2_negative] "m" (c2_negative),
	  [a_next]      "m" (a_next),
	  [b_next]      "m" (b_next)
	: "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11",
	  "v0","v1","v2","v3","v4","v5","v6","v7",
	  "v8","v9","v10","v11","v12","v13","v14","v15",
	  "v16","v17","v18","v19","v20","v21","v22","v23",
	  "v24","v25","v26","v27","v28","v29","v30","v31",
	  "cc","memory"
	);
}

#else

/*
 * Portable full-tile fallbacks keep the symbols linkable in non-AArch64
 * builds. The capability query above always rejects them; they are useful
 * only as a correctness backstop for an accidentally direct invocation.
 */
static void bls_sc_sgemm_12x8_2c_fallback
     (
       dim_t k,
       float alpha,
       const float* a,
       const float* b,
       float beta0,
       float* c0, inc_t rs0,
       float beta1,
       float* c1, inc_t rs1,
       int32_t relative_sign
     )
{
	for ( dim_t i = 0; i < 12; ++i )
	for ( dim_t j = 0; j < 8; ++j )
	{
		float rho = 0.0f;
		for ( dim_t p = 0; p < k; ++p )
			rho += a[p * 12 + i] * b[p * 8 + j];
		rho *= alpha;

		const float signed_rho = relative_sign > 0 ? rho : -rho;
		if ( beta0 == 0.0f ) c0[i * rs0 + j] = rho;
		else                 c0[i * rs0 + j] =
		                         beta0 * c0[i * rs0 + j] + rho;
		if ( beta1 == 0.0f ) c1[i * rs1 + j] = signed_rho;
		else                 c1[i * rs1 + j] =
		                         beta1 * c1[i * rs1 + j] + signed_rho;
	}
}

static void bls_sc_dgemm_8x6_2c_fallback
     (
       dim_t k,
       double alpha,
       const double* a,
       const double* b,
       double beta0,
       double* c0, inc_t rs0,
       double beta1,
       double* c1, inc_t rs1,
       int32_t relative_sign
     )
{
	for ( dim_t i = 0; i < 8; ++i )
	for ( dim_t j = 0; j < 6; ++j )
	{
		double rho = 0.0;
		for ( dim_t p = 0; p < k; ++p )
			rho += a[p * 8 + i] * b[p * 6 + j];
		rho *= alpha;

		const double signed_rho = relative_sign > 0 ? rho : -rho;
		if ( beta0 == 0.0 ) c0[i * rs0 + j] = rho;
		else                c0[i * rs0 + j] =
		                        beta0 * c0[i * rs0 + j] + rho;
		if ( beta1 == 0.0 ) c1[i * rs1 + j] = signed_rho;
		else                c1[i * rs1 + j] =
		                        beta1 * c1[i * rs1 + j] + signed_rho;
	}
}

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
     )
{
	const bls_sc_2c_params_t* p =
	    ( const bls_sc_2c_params_t* )bli_auxinfo_params( data );
	assert( m == 12 && n == 8 && cs_c == 1 && p != NULL );
	bls_sc_sgemm_12x8_2c_fallback
	(
	  k, *( const float* )alpha,
	  ( const float* )a, ( const float* )b,
	  *( const float* )beta, ( float* )c, rs_c,
	  *( const float* )p->beta2, ( float* )p->c2, p->rs_c2,
	  p->relative_sign
	);
	( void )cntx;
}

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
     )
{
	const bls_sc_2c_params_t* p =
	    ( const bls_sc_2c_params_t* )bli_auxinfo_params( data );
	assert( m == 8 && n == 6 && cs_c == 1 && p != NULL );
	bls_sc_dgemm_8x6_2c_fallback
	(
	  k, *( const double* )alpha,
	  ( const double* )a, ( const double* )b,
	  *( const double* )beta, ( double* )c, rs_c,
	  *( const double* )p->beta2, ( double* )p->c2, p->rs_c2,
	  p->relative_sign
	);
	( void )cntx;
}

#endif
