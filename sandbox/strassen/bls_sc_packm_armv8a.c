/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, BLIS contributors

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

#include "bls_sc_packm_armv8a.h"

#if defined(__aarch64__)

#include <arm_neon.h>

/* Match the scalar packer's exact expression and operand order. */
static inline float32x4_t bls_sc_combine_f32
     (
       float32x4_t x0,
       float32x4_t x1,
       int8_t      sign0,
       int8_t      sign1
     )
{
	if ( sign0 > 0 )
		return sign1 > 0 ? vaddq_f32( x0, x1 ) : vsubq_f32( x0, x1 );

	return sign1 > 0
	         ? vsubq_f32( x1, x0 )
	         : vsubq_f32( vnegq_f32( x0 ), x1 );
}

static inline float64x2_t bls_sc_combine_f64
     (
       float64x2_t x0,
       float64x2_t x1,
       int8_t      sign0,
       int8_t      sign1
     )
{
	if ( sign0 > 0 )
		return sign1 > 0 ? vaddq_f64( x0, x1 ) : vsubq_f64( x0, x1 );

	return sign1 > 0
	         ? vsubq_f64( x1, x0 )
	         : vsubq_f64( vnegq_f64( x0 ), x1 );
}

/*
 * Form one paired output from a shared vector. The unique pointer is allowed
 * to be NULL only for a singleton, and is not evaluated on that path.
 */
static inline float32x4_t bls_sc_pair_form_f32
     (
       float32x4_t  shared,
       const float* unique,
       inc_t        unique_offset,
       bool         singleton,
       int8_t       shared_sign,
       int8_t       unique_sign
     )
{
	if ( singleton )
		return shared_sign > 0 ? shared : vnegq_f32( shared );

	return bls_sc_combine_f32
	       (
	         shared, vld1q_f32( unique + unique_offset ),
	         shared_sign, unique_sign
	       );
}

static inline float64x2_t bls_sc_pair_form_f64
     (
       float64x2_t   shared,
       const double* unique,
       inc_t         unique_offset,
       bool          singleton,
       int8_t        shared_sign,
       int8_t        unique_sign
     )
{
	if ( singleton )
		return shared_sign > 0 ? shared : vnegq_f64( shared );

	return bls_sc_combine_f64
	       (
	         shared, vld1q_f64( unique + unique_offset ),
	         shared_sign, unique_sign
	       );
}

static inline float bls_sc_pair_form_scalar_f32
     (
       float        shared,
       const float* unique,
       inc_t        unique_offset,
       bool         singleton,
       int8_t       shared_sign,
       int8_t       unique_sign
     )
{
	if ( singleton ) return shared_sign > 0 ? shared : -shared;

	const float other = unique[unique_offset];

	if ( shared_sign > 0 )
		return unique_sign > 0 ? shared + other : shared - other;

	return unique_sign > 0 ? other - shared : -shared - other;
}

static inline double bls_sc_pair_form_scalar_f64
     (
       double        shared,
       const double* unique,
       inc_t         unique_offset,
       bool          singleton,
       int8_t        shared_sign,
       int8_t        unique_sign
     )
{
	if ( singleton ) return shared_sign > 0 ? shared : -shared;

	const double other = unique[unique_offset];

	if ( shared_sign > 0 )
		return unique_sign > 0 ? shared + other : shared - other;

	return unique_sign > 0 ? other - shared : -shared - other;
}

/* Transpose four combined rows and store four packed columns. */
static inline void bls_sc_transpose_store_f32_4x4
     (
       float32x4_t r0,
       float32x4_t r1,
       float32x4_t r2,
       float32x4_t r3,
       float*      p0,
       float*      p1,
       float*      p2,
       float*      p3
     )
{
	const float32x4_t t0 = vtrn1q_f32( r0, r1 );
	const float32x4_t t1 = vtrn2q_f32( r0, r1 );
	const float32x4_t t2 = vtrn1q_f32( r2, r3 );
	const float32x4_t t3 = vtrn2q_f32( r2, r3 );

	const float32x4_t c0 = vreinterpretq_f32_f64
	(
	  vtrn1q_f64( vreinterpretq_f64_f32( t0 ),
	              vreinterpretq_f64_f32( t2 ) )
	);
	const float32x4_t c1 = vreinterpretq_f32_f64
	(
	  vtrn1q_f64( vreinterpretq_f64_f32( t1 ),
	              vreinterpretq_f64_f32( t3 ) )
	);
	const float32x4_t c2 = vreinterpretq_f32_f64
	(
	  vtrn2q_f64( vreinterpretq_f64_f32( t0 ),
	              vreinterpretq_f64_f32( t2 ) )
	);
	const float32x4_t c3 = vreinterpretq_f32_f64
	(
	  vtrn2q_f64( vreinterpretq_f64_f32( t1 ),
	              vreinterpretq_f64_f32( t3 ) )
	);

	vst1q_f32( p0, c0 );
	vst1q_f32( p1, c1 );
	vst1q_f32( p2, c2 );
	vst1q_f32( p3, c3 );
}

/* Transpose two combined rows and store two packed columns. */
static inline void bls_sc_transpose_store_f64_2x2
     (
       float64x2_t r0,
       float64x2_t r1,
       double*     p0,
       double*     p1
     )
{
	vst1q_f64( p0, vtrn1q_f64( r0, r1 ) );
	vst1q_f64( p1, vtrn2q_f64( r0, r1 ) );
}

#define BLS_SC_F32_ADD(X0,X1) vaddq_f32( (X0), (X1) )
#define BLS_SC_F32_SUB(X0,X1) vsubq_f32( (X0), (X1) )
#define BLS_SC_F32_RSUB(X0,X1) vsubq_f32( (X1), (X0) )
#define BLS_SC_F32_NEGSUB(X0,X1) vsubq_f32( vnegq_f32( (X0) ), (X1) )
#define BLS_SC_F64_ADD(X0,X1) vaddq_f64( (X0), (X1) )
#define BLS_SC_F64_SUB(X0,X1) vsubq_f64( (X0), (X1) )
#define BLS_SC_F64_RSUB(X0,X1) vsubq_f64( (X1), (X0) )
#define BLS_SC_F64_NEGSUB(X0,X1) vsubq_f64( vnegq_f64( (X0) ), (X1) )

static void bls_sc_spackm_2src_rows
     (
             dim_t   k,
             dim_t   panel_dim,
       const float*  x0,
       const float*  x1,
             inc_t   cs_x,
             int8_t  sign0,
             int8_t  sign1,
             float*  p
     )
{
#define BLS_SC_SPACKM_2SRC_ROWS(OP) \
	do { \
		for ( dim_t l = 0; l < k; ++l ) \
		{ \
			const float* const a = x0 + l * cs_x; \
			const float* const b = x1 + l * cs_x; \
			float* const       d = p  + l * panel_dim; \
			for ( dim_t i = 0; i < panel_dim; i += 4 ) \
				vst1q_f32 \
				( \
				  d + i, OP( vld1q_f32( a + i ), vld1q_f32( b + i ) ) \
				); \
		} \
	} while ( 0 )

	if ( sign0 > 0 )
	{
		if ( sign1 > 0 ) BLS_SC_SPACKM_2SRC_ROWS( BLS_SC_F32_ADD );
		else             BLS_SC_SPACKM_2SRC_ROWS( BLS_SC_F32_SUB );
	}
	else
	{
		if ( sign1 > 0 ) BLS_SC_SPACKM_2SRC_ROWS( BLS_SC_F32_RSUB );
		else             BLS_SC_SPACKM_2SRC_ROWS( BLS_SC_F32_NEGSUB );
	}

#undef BLS_SC_SPACKM_2SRC_ROWS
}

static void bls_sc_spackm_2src_cols
     (
             dim_t   k,
             dim_t   panel_dim,
       const float*  x0,
       const float*  x1,
             inc_t   rs_x,
             int8_t  sign0,
             int8_t  sign1,
             float*  p
     )
{
	dim_t l = 0;
#define BLS_SC_SPACKM_2SRC_COLS(OP) \
	do { \
		for ( ; l + 4 <= k; l += 4 ) \
		for ( dim_t i = 0; i < panel_dim; i += 4 ) \
		{ \
			const float32x4_t r0 = OP \
			( \
			  vld1q_f32( x0 + ( i + 0 ) * rs_x + l ), \
			  vld1q_f32( x1 + ( i + 0 ) * rs_x + l ) \
			); \
			const float32x4_t r1 = OP \
			( \
			  vld1q_f32( x0 + ( i + 1 ) * rs_x + l ), \
			  vld1q_f32( x1 + ( i + 1 ) * rs_x + l ) \
			); \
			const float32x4_t r2 = OP \
			( \
			  vld1q_f32( x0 + ( i + 2 ) * rs_x + l ), \
			  vld1q_f32( x1 + ( i + 2 ) * rs_x + l ) \
			); \
			const float32x4_t r3 = OP \
			( \
			  vld1q_f32( x0 + ( i + 3 ) * rs_x + l ), \
			  vld1q_f32( x1 + ( i + 3 ) * rs_x + l ) \
			); \
			bls_sc_transpose_store_f32_4x4 \
			( \
			  r0, r1, r2, r3, \
			  p + ( l + 0 ) * panel_dim + i, \
			  p + ( l + 1 ) * panel_dim + i, \
			  p + ( l + 2 ) * panel_dim + i, \
			  p + ( l + 3 ) * panel_dim + i \
			); \
		} \
	} while ( 0 )

	if ( sign0 > 0 )
	{
		if ( sign1 > 0 ) BLS_SC_SPACKM_2SRC_COLS( BLS_SC_F32_ADD );
		else             BLS_SC_SPACKM_2SRC_COLS( BLS_SC_F32_SUB );
	}
	else
	{
		if ( sign1 > 0 ) BLS_SC_SPACKM_2SRC_COLS( BLS_SC_F32_RSUB );
		else             BLS_SC_SPACKM_2SRC_COLS( BLS_SC_F32_NEGSUB );
	}

#undef BLS_SC_SPACKM_2SRC_COLS

	/* The transpose path consumes four K columns at a time. */
	for ( ; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; ++i )
		{
			const float a = x0[i * rs_x + l];
			const float b = x1[i * rs_x + l];

			if ( sign0 > 0 )
				p[l * panel_dim + i] = sign1 > 0 ? a + b : a - b;
			else
				p[l * panel_dim + i] = sign1 > 0 ? b - a : -a - b;
		}
}

static void bls_sc_dpackm_2src_rows
     (
             dim_t    k,
             dim_t    panel_dim,
       const double*  x0,
       const double*  x1,
             inc_t    cs_x,
             int8_t   sign0,
             int8_t   sign1,
             double*  p
     )
{
#define BLS_SC_DPACKM_2SRC_ROWS(OP) \
	do { \
		for ( dim_t l = 0; l < k; ++l ) \
		{ \
			const double* const a = x0 + l * cs_x; \
			const double* const b = x1 + l * cs_x; \
			double* const       d = p  + l * panel_dim; \
			for ( dim_t i = 0; i < panel_dim; i += 2 ) \
				vst1q_f64 \
				( \
				  d + i, OP( vld1q_f64( a + i ), vld1q_f64( b + i ) ) \
				); \
		} \
	} while ( 0 )

	if ( sign0 > 0 )
	{
		if ( sign1 > 0 ) BLS_SC_DPACKM_2SRC_ROWS( BLS_SC_F64_ADD );
		else             BLS_SC_DPACKM_2SRC_ROWS( BLS_SC_F64_SUB );
	}
	else
	{
		if ( sign1 > 0 ) BLS_SC_DPACKM_2SRC_ROWS( BLS_SC_F64_RSUB );
		else             BLS_SC_DPACKM_2SRC_ROWS( BLS_SC_F64_NEGSUB );
	}

#undef BLS_SC_DPACKM_2SRC_ROWS
}

static void bls_sc_dpackm_2src_cols
     (
             dim_t    k,
             dim_t    panel_dim,
       const double*  x0,
       const double*  x1,
             inc_t    rs_x,
             int8_t   sign0,
             int8_t   sign1,
             double*  p
     )
{
	dim_t l = 0;
#define BLS_SC_DPACKM_2SRC_COLS(OP) \
	do { \
		for ( ; l + 2 <= k; l += 2 ) \
		for ( dim_t i = 0; i < panel_dim; i += 2 ) \
		{ \
			const float64x2_t r0 = OP \
			( \
			  vld1q_f64( x0 + ( i + 0 ) * rs_x + l ), \
			  vld1q_f64( x1 + ( i + 0 ) * rs_x + l ) \
			); \
			const float64x2_t r1 = OP \
			( \
			  vld1q_f64( x0 + ( i + 1 ) * rs_x + l ), \
			  vld1q_f64( x1 + ( i + 1 ) * rs_x + l ) \
			); \
			bls_sc_transpose_store_f64_2x2 \
			( \
			  r0, r1, \
			  p + ( l + 0 ) * panel_dim + i, \
			  p + ( l + 1 ) * panel_dim + i \
			); \
		} \
	} while ( 0 )

	if ( sign0 > 0 )
	{
		if ( sign1 > 0 ) BLS_SC_DPACKM_2SRC_COLS( BLS_SC_F64_ADD );
		else             BLS_SC_DPACKM_2SRC_COLS( BLS_SC_F64_SUB );
	}
	else
	{
		if ( sign1 > 0 ) BLS_SC_DPACKM_2SRC_COLS( BLS_SC_F64_RSUB );
		else             BLS_SC_DPACKM_2SRC_COLS( BLS_SC_F64_NEGSUB );
	}

#undef BLS_SC_DPACKM_2SRC_COLS

	/* The transpose path consumes two K columns at a time. */
	for ( ; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; ++i )
		{
			const double a = x0[i * rs_x + l];
			const double b = x1[i * rs_x + l];

			if ( sign0 > 0 )
				p[l * panel_dim + i] = sign1 > 0 ? a + b : a - b;
			else
				p[l * panel_dim + i] = sign1 > 0 ? b - a : -a - b;
		}
}

#undef BLS_SC_F32_ADD
#undef BLS_SC_F32_SUB
#undef BLS_SC_F32_RSUB
#undef BLS_SC_F32_NEGSUB
#undef BLS_SC_F64_ADD
#undef BLS_SC_F64_SUB
#undef BLS_SC_F64_RSUB
#undef BLS_SC_F64_NEGSUB

static void bls_sc_spackm_pair_rows
     (
             dim_t         k,
             dim_t         panel_dim,
       const float*        shared,
       const float* const  unique[2],
             inc_t         cs_x,
       const bool          singleton[2],
       const int8_t        shared_sign[2],
       const int8_t        unique_sign[2],
             float* const  p[2]
     )
{
	for ( dim_t l = 0; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; i += 4 )
		{
			const inc_t off = l * cs_x + i;
			const float32x4_t s = vld1q_f32( shared + off );

			const float32x4_t v0 = bls_sc_pair_form_f32
			(
			  s, unique[0], off, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float32x4_t v1 = bls_sc_pair_form_f32
			(
			  s, unique[1], off, singleton[1],
			  shared_sign[1], unique_sign[1]
			);

			vst1q_f32( p[0] + l * panel_dim + i, v0 );
			vst1q_f32( p[1] + l * panel_dim + i, v1 );
		}
}

static void bls_sc_spackm_pair_cols
     (
             dim_t         k,
             dim_t         panel_dim,
       const float*        shared,
       const float* const  unique[2],
             inc_t         rs_x,
       const bool          singleton[2],
       const int8_t        shared_sign[2],
       const int8_t        unique_sign[2],
             float* const  p[2]
     )
{
	dim_t l = 0;
	for ( ; l + 4 <= k; l += 4 )
		for ( dim_t i = 0; i < panel_dim; i += 4 )
		{
			const inc_t off0 = ( i + 0 ) * rs_x + l;
			const inc_t off1 = ( i + 1 ) * rs_x + l;
			const inc_t off2 = ( i + 2 ) * rs_x + l;
			const inc_t off3 = ( i + 3 ) * rs_x + l;
			const float32x4_t s0 = vld1q_f32( shared + off0 );
			const float32x4_t s1 = vld1q_f32( shared + off1 );
			const float32x4_t s2 = vld1q_f32( shared + off2 );
			const float32x4_t s3 = vld1q_f32( shared + off3 );

			const float32x4_t r00 = bls_sc_pair_form_f32
			(
			  s0, unique[0], off0, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float32x4_t r01 = bls_sc_pair_form_f32
			(
			  s1, unique[0], off1, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float32x4_t r02 = bls_sc_pair_form_f32
			(
			  s2, unique[0], off2, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float32x4_t r03 = bls_sc_pair_form_f32
			(
			  s3, unique[0], off3, singleton[0],
			  shared_sign[0], unique_sign[0]
			);

			bls_sc_transpose_store_f32_4x4
			(
			  r00, r01, r02, r03,
			  p[0] + ( l + 0 ) * panel_dim + i,
			  p[0] + ( l + 1 ) * panel_dim + i,
			  p[0] + ( l + 2 ) * panel_dim + i,
			  p[0] + ( l + 3 ) * panel_dim + i
			);

			const float32x4_t r10 = bls_sc_pair_form_f32
			(
			  s0, unique[1], off0, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
			const float32x4_t r11 = bls_sc_pair_form_f32
			(
			  s1, unique[1], off1, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
			const float32x4_t r12 = bls_sc_pair_form_f32
			(
			  s2, unique[1], off2, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
			const float32x4_t r13 = bls_sc_pair_form_f32
			(
			  s3, unique[1], off3, singleton[1],
			  shared_sign[1], unique_sign[1]
			);

			bls_sc_transpose_store_f32_4x4
			(
			  r10, r11, r12, r13,
			  p[1] + ( l + 0 ) * panel_dim + i,
			  p[1] + ( l + 1 ) * panel_dim + i,
			  p[1] + ( l + 2 ) * panel_dim + i,
			  p[1] + ( l + 3 ) * panel_dim + i
			);
		}

	/* Scalar K tails retain the same one-load shared-source property. */
	for ( ; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; ++i )
		{
			const inc_t off = i * rs_x + l;
			const float   s = shared[off];

			p[0][l * panel_dim + i] = bls_sc_pair_form_scalar_f32
			(
			  s, unique[0], off, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			p[1][l * panel_dim + i] = bls_sc_pair_form_scalar_f32
			(
			  s, unique[1], off, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
		}
}

static void bls_sc_dpackm_pair_rows
     (
             dim_t          k,
             dim_t          panel_dim,
       const double*        shared,
       const double* const  unique[2],
             inc_t          cs_x,
       const bool           singleton[2],
       const int8_t         shared_sign[2],
       const int8_t         unique_sign[2],
             double* const  p[2]
     )
{
	for ( dim_t l = 0; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; i += 2 )
		{
			const inc_t off = l * cs_x + i;
			const float64x2_t s = vld1q_f64( shared + off );

			const float64x2_t v0 = bls_sc_pair_form_f64
			(
			  s, unique[0], off, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float64x2_t v1 = bls_sc_pair_form_f64
			(
			  s, unique[1], off, singleton[1],
			  shared_sign[1], unique_sign[1]
			);

			vst1q_f64( p[0] + l * panel_dim + i, v0 );
			vst1q_f64( p[1] + l * panel_dim + i, v1 );
		}
}

static void bls_sc_dpackm_pair_cols
     (
             dim_t          k,
             dim_t          panel_dim,
       const double*        shared,
       const double* const  unique[2],
             inc_t          rs_x,
       const bool           singleton[2],
       const int8_t         shared_sign[2],
       const int8_t         unique_sign[2],
             double* const  p[2]
     )
{
	dim_t l = 0;
	for ( ; l + 2 <= k; l += 2 )
		for ( dim_t i = 0; i < panel_dim; i += 2 )
		{
			const inc_t off0 = ( i + 0 ) * rs_x + l;
			const inc_t off1 = ( i + 1 ) * rs_x + l;
			const float64x2_t s0 = vld1q_f64( shared + off0 );
			const float64x2_t s1 = vld1q_f64( shared + off1 );

			const float64x2_t r00 = bls_sc_pair_form_f64
			(
			  s0, unique[0], off0, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			const float64x2_t r01 = bls_sc_pair_form_f64
			(
			  s1, unique[0], off1, singleton[0],
			  shared_sign[0], unique_sign[0]
			);

			bls_sc_transpose_store_f64_2x2
			(
			  r00, r01,
			  p[0] + ( l + 0 ) * panel_dim + i,
			  p[0] + ( l + 1 ) * panel_dim + i
			);

			const float64x2_t r10 = bls_sc_pair_form_f64
			(
			  s0, unique[1], off0, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
			const float64x2_t r11 = bls_sc_pair_form_f64
			(
			  s1, unique[1], off1, singleton[1],
			  shared_sign[1], unique_sign[1]
			);

			bls_sc_transpose_store_f64_2x2
			(
			  r10, r11,
			  p[1] + ( l + 0 ) * panel_dim + i,
			  p[1] + ( l + 1 ) * panel_dim + i
			);
		}

	/* Scalar K tails retain the same one-load shared-source property. */
	for ( ; l < k; ++l )
		for ( dim_t i = 0; i < panel_dim; ++i )
		{
			const inc_t off = i * rs_x + l;
			const double  s = shared[off];

			p[0][l * panel_dim + i] = bls_sc_pair_form_scalar_f64
			(
			  s, unique[0], off, singleton[0],
			  shared_sign[0], unique_sign[0]
			);
			p[1][l * panel_dim + i] = bls_sc_pair_form_scalar_f64
			(
			  s, unique[1], off, singleton[1],
			  shared_sign[1], unique_sign[1]
			);
		}
}

#endif

bool bls_sc_packm_armv8a_is_supported
     (
       num_t dt,
       dim_t panel_m,
       dim_t panel_dim,
       inc_t rs_x,
       inc_t cs_x
     )
{
#if defined(__aarch64__)
	if ( panel_m != panel_dim || panel_dim <= 0 ||
	     ( rs_x != 1 && cs_x != 1 ) )
		return FALSE;

	return ( dt == BLIS_FLOAT &&
	         ( panel_dim == 8 || panel_dim == 12 ) ) ||
	       ( dt == BLIS_DOUBLE &&
	         ( panel_dim == 6 || panel_dim == 8 ) );
#else
	( void )dt;
	( void )panel_m;
	( void )panel_dim;
	( void )rs_x;
	( void )cs_x;
	return FALSE;
#endif
}

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
     )
{
#if defined(__aarch64__)
	if ( !bls_sc_packm_armv8a_is_supported
	      ( dt, panel_m, panel_dim, rs_x, cs_x ) ||
	     k < 0 ||
	     ( sign0 != 1 && sign0 != -1 ) ||
	     ( sign1 != 1 && sign1 != -1 ) )
		return FALSE;

	if ( k > 0 && ( x0 == NULL || x1 == NULL || p == NULL ) )
		return FALSE;

	if ( dt == BLIS_FLOAT )
	{
		if ( rs_x == 1 )
			bls_sc_spackm_2src_rows
			(
			  k, panel_dim,
			  ( const float* )x0, ( const float* )x1, cs_x,
			  sign0, sign1, ( float* )p
			);
		else
			bls_sc_spackm_2src_cols
			(
			  k, panel_dim,
			  ( const float* )x0, ( const float* )x1, rs_x,
			  sign0, sign1, ( float* )p
			);

		return TRUE;
	}

	if ( dt == BLIS_DOUBLE )
	{
		if ( rs_x == 1 )
			bls_sc_dpackm_2src_rows
			(
			  k, panel_dim,
			  ( const double* )x0, ( const double* )x1, cs_x,
			  sign0, sign1, ( double* )p
			);
		else
			bls_sc_dpackm_2src_cols
			(
			  k, panel_dim,
			  ( const double* )x0, ( const double* )x1, rs_x,
			  sign0, sign1, ( double* )p
			);

		return TRUE;
	}
#else
	( void )dt;
	( void )panel_m;
	( void )k;
	( void )panel_dim;
	( void )x0;
	( void )x1;
	( void )rs_x;
	( void )cs_x;
	( void )sign0;
	( void )sign1;
	( void )p;
#endif

	return FALSE;
}

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
     )
{
#if defined(__aarch64__)
	if ( unique == NULL || shared_sign == NULL ||
	     unique_sign == NULL || p == NULL ||
	     !bls_sc_packm_armv8a_is_supported
	      ( dt, panel_m, panel_dim, rs_x, cs_x ) ||
	     k < 0 )
		return FALSE;

	bool singleton[2];
	for ( dim_t f = 0; f < 2; ++f )
	{
		if ( shared_sign[f] != 1 && shared_sign[f] != -1 )
			return FALSE;

		singleton[f] = unique[f] == NULL && unique_sign[f] == 0;
		if ( !singleton[f] &&
		     ( unique[f] == NULL ||
		       ( unique_sign[f] != 1 && unique_sign[f] != -1 ) ) )
			return FALSE;
	}

	if ( k > 0 && ( shared == NULL || p[0] == NULL || p[1] == NULL ) )
		return FALSE;

	if ( dt == BLIS_FLOAT )
	{
		const float* unique_s[2] =
		{
			( const float* )unique[0],
			( const float* )unique[1]
		};
		float* p_s[2] = { ( float* )p[0], ( float* )p[1] };

		if ( rs_x == 1 )
			bls_sc_spackm_pair_rows
			(
			  k, panel_dim, ( const float* )shared,
			  unique_s, cs_x, singleton,
			  shared_sign, unique_sign, p_s
			);
		else
			bls_sc_spackm_pair_cols
			(
			  k, panel_dim, ( const float* )shared,
			  unique_s, rs_x, singleton,
			  shared_sign, unique_sign, p_s
			);

		return TRUE;
	}

	if ( dt == BLIS_DOUBLE )
	{
		const double* unique_d[2] =
		{
			( const double* )unique[0],
			( const double* )unique[1]
		};
		double* p_d[2] = { ( double* )p[0], ( double* )p[1] };

		if ( rs_x == 1 )
			bls_sc_dpackm_pair_rows
			(
			  k, panel_dim, ( const double* )shared,
			  unique_d, cs_x, singleton,
			  shared_sign, unique_sign, p_d
			);
		else
			bls_sc_dpackm_pair_cols
			(
			  k, panel_dim, ( const double* )shared,
			  unique_d, rs_x, singleton,
			  shared_sign, unique_sign, p_d
			);

		return TRUE;
	}
#else
	( void )dt;
	( void )panel_m;
	( void )k;
	( void )panel_dim;
	( void )shared;
	( void )unique;
	( void )rs_x;
	( void )cs_x;
	( void )shared_sign;
	( void )unique_sign;
	( void )p;
#endif

	return FALSE;
}
