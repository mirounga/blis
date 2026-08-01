#ifndef BLS_SC_PACKM_ARMV8A_H
#define BLS_SC_PACKM_ARMV8A_H

#include "blis.h"

#include <stdint.h>

/* Query the full-panel datatype/layout subset handled by this module. */
bool bls_sc_packm_armv8a_is_supported
     (
       num_t dt,
       dim_t panel_m,
       dim_t panel_dim,
       inc_t rs_x,
       inc_t cs_x
     );

/*
 * Try to pack one full micropanel of the two-source linear form
 *
 *   sign0 * x0 + sign1 * x1,
 *
 * where sign0 and sign1 must each be +1 or -1. The two sources share the
 * same row and column strides. The destination has unit row stride and
 * column stride panel_dim, as expected by the native GEMM microkernels.
 *
 * This fast path supports real single and double precision only. It writes
 * exactly panel_dim * k elements; the caller remains responsible for K
 * padding and any padding implied by the packed-panel stride. A false return
 * means that the caller must use its portable packing path. In particular,
 * short edge panels and general-stride sources are deliberately declined.
 */
bool bls_sc_packm_2src_armv8a
     (
             num_t   dt,
             dim_t   panel_m,
             dim_t   k,
             dim_t   panel_dim,
       const void*   x0,
       const void*   x1,
             inc_t   rs_x,
             inc_t   cs_x,
             int8_t  sign0,
             int8_t  sign1,
             void*   p
     );

/*
 * Try to pack two forms that share one source:
 *
 *   p[f] = shared_sign[f] * shared + unique_sign[f] * unique[f].
 *
 * An active unique source has a coefficient of +1 or -1. A NULL unique
 * source paired with a zero coefficient denotes a singleton form containing
 * only its signed shared source. Both forms must be described, even when one
 * or both are singletons. The source strides and packing restrictions match
 * bls_sc_packm_2src_armv8a(). The two destination regions must not overlap.
 */
bool bls_sc_packm_pair_armv8a
     (
             num_t         dt,
             dim_t         panel_m,
             dim_t         k,
             dim_t         panel_dim,
       const void*         shared,
       const void* const   unique[2],
             inc_t         rs_x,
             inc_t         cs_x,
       const int8_t        shared_sign[2],
       const int8_t        unique_sign[2],
             void* const   p[2]
     );

#endif
