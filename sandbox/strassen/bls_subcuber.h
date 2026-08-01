#ifndef BLS_SUBCUBER_H
#define BLS_SUBCUBER_H

void bls_subcuber_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     );

#endif
