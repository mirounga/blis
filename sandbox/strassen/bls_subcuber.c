#include "blis.h"
#include "bls_sc_2c_armv8a.h"
#include "bls_sc_packm_armv8a.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLS_SC_NQUADS 4
#define BLS_SC_NTERMS 7
#define BLS_SC_NTHREAD_LEVELS 7
#define BLS_SC_BATCH_CAPACITY 2
#define BLS_SC_NBATCHES 4

typedef struct
{
	int8_t a[BLS_SC_NQUADS];
	int8_t b[BLS_SC_NQUADS];
	int8_t c[BLS_SC_NQUADS];
} bls_sc_term_t;

typedef struct
{
	thrinfo_t* jc;
	thrinfo_t* pc;
	thrinfo_t* pb;
	thrinfo_t* ic;
	thrinfo_t* pa;
	thrinfo_t* jr;
	thrinfo_t* ir;
} bls_sc_thrinfo_path_t;

typedef struct
{
	dim_t n_terms;
	dim_t term[BLS_SC_BATCH_CAPACITY];
} bls_sc_batch_t;

typedef struct
{
	dim_t  shared;
	dim_t  unique[BLS_SC_BATCH_CAPACITY];
	int8_t shared_coeff[BLS_SC_BATCH_CAPACITY];
	int8_t unique_coeff[BLS_SC_BATCH_CAPACITY];
} bls_sc_pair_plan_t;

typedef struct
{
	dim_t   term;
	uint8_t init_mask;
	dim_t   direct_q;
	dim_t   fanout_q;
	dim_t   two_q0;
	dim_t   two_q1;
} bls_sc_term_exec_t;

typedef struct
{
	num_t        dt;
	dim_t        mr;
	dim_t        nr;
	inc_t        rs_c;
	inc_t        cs_c;
	inc_t        rs_t;
	inc_t        cs_t;
	const void*  alpha;
	const void*  beta;
	const void*  zero;
	const void*  one;
	gemm_ukr_ft  gemm_ukr;
	gemm_ukr_ft  gemm_ukr_2c;
	const cntx_t* cntx;
} bls_sc_ukr_ctx_t;

/*
 * The one-level rank-7 Strassen tensor used by the fused SubCuber path.
 * Compared with the Winograd DAG, this form has lower expanded support:
 * every packed input is either one source or a two-source sum/difference,
 * and every product fans out to at most two output quadrants.
 *
 * Quadrant order is 00, 01, 10, 11. Each row describes
 *
 *   M_r = (sum_q a[q] A_q) (sum_q b[q] B_q)
 *   C_q += c[q] M_r.
 */
static const bls_sc_term_t bls_sc_terms[BLS_SC_NTERMS] =
{
	{ { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 } },
	{ { 0, 0, 1, 1 }, { 1, 0, 0, 0 }, { 0, 0, 1,-1 } },
	{ { 1, 0, 0, 0 }, { 0, 1, 0,-1 }, { 0, 1, 0, 1 } },
	{ { 0, 0, 0, 1 }, {-1, 0, 1, 0 }, { 1, 0, 1, 0 } },
	{ { 1, 1, 0, 0 }, { 0, 0, 0, 1 }, {-1, 1, 0, 0 } },
	{ {-1, 0, 1, 0 }, { 1, 1, 0, 0 }, { 0, 0, 0, 1 } },
	{ { 0, 1, 0,-1 }, { 0, 0, 1, 1 }, { 1, 0, 0, 0 } },
};

/*
 * Seed each output quadrant before any additive fanout reaches it. This
 * folds beta into the first contribution and avoids a separate C pass.
 */
static const dim_t bls_sc_term_order_default[BLS_SC_NTERMS] =
    { 5, 6, 1, 2, 0, 3, 4 };
static const dim_t bls_sc_init_term_default[BLS_SC_NQUADS] =
    { 6, 2, 1, 5 };

/*
 * Experimental locality trail: every consecutive pair shares at least one
 * output quadrant. All first contributions remain positive, so beta can
 * still be folded into the first touch without a separate scaling pass.
 */
static const dim_t bls_sc_term_order_trail[BLS_SC_NTERMS] =
    { 5, 1, 3, 6, 4, 2, 0 };
static const dim_t bls_sc_init_term_trail[BLS_SC_NQUADS] =
    { 3, 4, 1, 5 };

/*
 * Experimental paired schedule. Each two-form batch shares one A source,
 * one B source, and one C destination. Keeping both products at the same C
 * microtile also amortizes the A/B pack publication barriers over the pair.
 */
static const bls_sc_batch_t bls_sc_batches[BLS_SC_NBATCHES] =
{
	{ 1, { 5, 0 } },
	{ 2, { 6, 0 } },
	{ 2, { 1, 3 } },
	{ 2, { 2, 4 } },
};

static err_t bls_sc_gemm_int
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     );

static void bls_sc_gemm_bp
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     );

static bool bls_sc_env_enabled( void )
{
	const char* disable = getenv( "BLIS_SUBCUBER_DISABLE" );

	return disable == NULL || disable[0] == '\0' || disable[0] == '0';
}

static bool bls_sc_env_flag
     (
       const char* name
     )
{
	const char* value = getenv( name );

	return value != NULL && value[0] != '\0' && value[0] != '0';
}

static bool bls_sc_simd_pack_enabled( void )
{
#if defined(__aarch64__)
	return !bls_sc_env_flag( "BLIS_SUBCUBER_DISABLE_SIMD_PACK" );
#else
	return FALSE;
#endif
}

static bool bls_sc_simd_pack_supported
     (
       num_t dt,
       dim_t panel_dim,
       inc_t rs_src,
       inc_t cs_src
     )
{
	return bls_sc_simd_pack_enabled() &&
	       bls_sc_packm_armv8a_is_supported
	       ( dt, panel_dim, panel_dim, rs_src, cs_src );
}

static bool bls_sc_pair_simd_pack_supported
     (
       num_t dt,
       dim_t m,
       dim_t panel_dim,
       inc_t rs_src,
       inc_t cs_src
     )
{
	return !bls_sc_env_flag( "BLIS_SUBCUBER_DISABLE_PAIR_FUSED_PACK" ) &&
	       panel_dim > 0 && m % panel_dim == 0 &&
	       bls_sc_simd_pack_supported( dt, panel_dim, rs_src, cs_src );
}

static bool bls_sc_size_add
     (
       siz_t  a,
       siz_t  b,
       siz_t* sum
     )
{
	if ( a > ( siz_t )-1 - b ) return FALSE;

	*sum = a + b;
	return TRUE;
}

static bool bls_sc_size_mul
     (
       siz_t  a,
       siz_t  b,
       siz_t* product
     )
{
	if ( a != 0 && b > ( siz_t )-1 / a ) return FALSE;

	*product = a * b;
	return TRUE;
}

static bool bls_sc_size_round_up
     (
       siz_t  value,
       siz_t  multiple,
       siz_t* rounded
     )
{
	if ( multiple == 0 ) return FALSE;

	const siz_t remainder = value % multiple;
	const siz_t increment = remainder == 0 ? 0 : multiple - remainder;

	return bls_sc_size_add( value, increment, rounded );
}

static void bls_sc_pack_size_abort
     (
       const thrinfo_t* thread
     )
{
	if ( thread == NULL || bli_thrinfo_am_chief( thread ) )
		bli_print_msg
		(
		  "BLIS SubCuber pack-buffer size overflow",
		  __FILE__,
		  __LINE__
		);

	bli_abort();
}

static dim_t bls_sc_env_blksz
     (
       const char* name,
             dim_t fallback,
             dim_t multiple
     )
{
	const char* text = getenv( name );
	if ( text == NULL || text[0] == '\0' ) return fallback;

	char* end = NULL;
	const unsigned long long parsed = strtoull( text, &end, 10 );
	if ( end == text || *end != '\0' || parsed == 0 ) return fallback;

	dim_t value = ( dim_t )parsed;
	if ( value <= 0 ) return fallback;

	value = ( value / multiple ) * multiple;
	return value >= multiple ? value : fallback;
}

static dim_t bls_sc_default_crossover
     (
       num_t dt,
       dim_t nt,
       bool  can_use_2c
     )
{
	/*
	 * Conservative Firestorm crossovers measured with the same paired
	 * harness used for the performance table. Portable scratch/scatter
	 * contexts use a higher floor; every value remains overrideable.
	 */
	if ( !can_use_2c ) return 2048;

	if ( dt == BLIS_FLOAT )
		return nt <= 4 ? 768 : 2304;

	if ( nt <= 1 ) return 384;
	if ( nt <= 4 ) return 768;
	return 1536;
}

static bool bls_sc_operands_supported
     (
       const obj_t* a,
       const obj_t* b,
       const obj_t* c
     )
{
	const num_t dt = bli_obj_dt( c );

	if ( dt != BLIS_FLOAT && dt != BLIS_DOUBLE ) return FALSE;
	if ( bli_obj_dt( a ) != dt || bli_obj_dt( b ) != dt ) return FALSE;
	if ( bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) return FALSE;

	if ( !bli_obj_is_general( a ) ||
	     !bli_obj_is_general( b ) ||
	     !bli_obj_is_general( c ) ) return FALSE;

	if ( !bli_obj_is_dense( a ) ||
	     !bli_obj_is_dense( b ) ||
	     !bli_obj_is_dense( c ) ) return FALSE;

	/*
	 * The conventional control tree absorbs attached matrix scalars into
	 * packing. This prototype deliberately falls back until its generated
	 * linear-combination packer carries equivalent scalar metadata.
	 */
	if ( !bli_obj_scalar_equals( a, &BLIS_ONE ) ||
	     !bli_obj_scalar_equals( b, &BLIS_ONE ) ||
	     !bli_obj_scalar_equals( c, &BLIS_ONE ) ) return FALSE;

	return TRUE;
}

/*
 * SubCuber uses the full jc->pc->pb->ic->pa->jr->ir SUP thread tree. Keep
 * this validation unconditional: it runs once per worker invocation, makes a
 * future decorator/tree mismatch fail deterministically, and also verifies
 * that the n_way factors consume the complete launched team.
 */
static bool bls_sc_thrinfo_path_init
     (
             thrinfo_t*             root,
       const rntm_t*                rntm,
             bls_sc_thrinfo_path_t* path
     )
{
	if ( root == NULL || rntm == NULL ) return FALSE;

	path->jc = bli_thrinfo_sub_node( 0, root );
	if ( path->jc == NULL ) return FALSE;
	path->pc = bli_thrinfo_sub_node( 0, path->jc );
	if ( path->pc == NULL ) return FALSE;
	path->pb = bli_thrinfo_sub_node( 0, path->pc );
	if ( path->pb == NULL ) return FALSE;
	path->ic = bli_thrinfo_sub_node( 0, path->pb );
	if ( path->ic == NULL ) return FALSE;
	path->pa = bli_thrinfo_sub_node( 0, path->ic );
	if ( path->pa == NULL ) return FALSE;
	path->jr = bli_thrinfo_sub_node( 0, path->pa );
	if ( path->jr == NULL ) return FALSE;
	path->ir = bli_thrinfo_sub_node( 0, path->jr );
	if ( path->ir == NULL ) return FALSE;

	thrinfo_t* nodes[BLS_SC_NTHREAD_LEVELS] =
	{
		path->jc, path->pc, path->pb, path->ic,
		path->pa, path->jr, path->ir
	};
	const dim_t expected_ways[BLS_SC_NTHREAD_LEVELS] =
	{
		bli_rntm_ways_for( BLIS_NC, rntm ),
		bli_rntm_ways_for( BLIS_KC, rntm ),
		1,
		bli_rntm_ways_for( BLIS_MC, rntm ),
		1,
		bli_rntm_ways_for( BLIS_NR, rntm ),
		bli_rntm_ways_for( BLIS_MR, rntm )
	};
	dim_t parent_nt = bli_thrinfo_num_threads( root );
	dim_t way_product = 1;

	if ( parent_nt != bli_rntm_num_threads( rntm ) ) return FALSE;

	for ( dim_t i = 0; i < BLS_SC_NTHREAD_LEVELS; ++i )
	{
		const dim_t way = bli_thrinfo_n_way( nodes[i] );

		if ( way != expected_ways[i] ||
		     way < 1 ||
		     parent_nt % way != 0 ||
		     bli_thrinfo_num_threads( nodes[i] ) != parent_nt / way )
			return FALSE;

		if ( way_product > bli_thrinfo_num_threads( root ) / way )
			return FALSE;

		way_product *= way;
		parent_nt /= way;
	}

	return parent_nt == 1 &&
	       way_product == bli_thrinfo_num_threads( root );
}

static void bls_sc_thrinfo_path_require
     (
             thrinfo_t*             root,
       const rntm_t*                rntm,
             bls_sc_thrinfo_path_t* path
     )
{
	if ( bls_sc_thrinfo_path_init( root, rntm, path ) ) return;

	if ( root == NULL || bli_thrinfo_am_chief( root ) )
		bli_print_msg
		(
		  "BLIS SubCuber requires the full "
		  "jc->pc->pb->ic->pa->jr->ir SUP thread tree",
		  __FILE__,
		  __LINE__
		);

	bli_abort();
}

static void bls_sc_default_gemm
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	const rntm_t* rntm_use = rntm;
	rntm_t rntm_safe;

	/*
	 * This revision's conventional DGEMM schedule is also unsafe when a
	 * caller explicitly splits the IR loop. Make every fallback safe,
	 * including BLIS_SUBCUBER_DISABLE, by preserving the total team size
	 * while allowing the decorator to refactorize it across supported ways.
	 */
	if ( bli_obj_dt( c ) == BLIS_DOUBLE )
	{
		if ( rntm == NULL ) bli_rntm_init_from_global( &rntm_safe );
		else                rntm_safe = *rntm;
		rntm_use = &rntm_safe;

		if ( bli_rntm_ir_ways( &rntm_safe ) > 1 )
		{
			const dim_t requested_nt =
			    bli_rntm_calc_num_threads( &rntm_safe );
			bli_rntm_set_num_threads( requested_nt, &rntm_safe );
		}
	}

	bli_gemm_def_ex( alpha, a, b, beta, c, cntx, rntm_use );
}

static char* bls_sc_ptr_offset
     (
             void* p,
             inc_t row,
             inc_t col,
             inc_t rs,
             inc_t cs,
             dim_t dt_size
     )
{
	const inc_t off = row * rs + col * cs;
	return ( char* )p + off * ( inc_t )dt_size;
}

static const char* bls_sc_const_ptr_offset
     (
       const void* p,
             inc_t row,
             inc_t col,
             inc_t rs,
             inc_t cs,
             dim_t dt_size
     )
{
	const inc_t off = row * rs + col * cs;
	return ( const char* )p + off * ( inc_t )dt_size;
}

#define BLS_SC_PACK_PANEL( expr ) \
do \
{ \
	for ( dim_t l = 0; l < k; ++l ) \
	{ \
		dim_t i = 0; \
		for ( ; i < panel_m; ++i ) \
		{ \
			const inc_t off = i * rs_src + l * cs_src; \
			dst[l * panel_dim_max + i] = ( expr ); \
		} \
		for ( ; i < panel_dim_max; ++i ) \
			dst[l * panel_dim_max + i] = 0; \
	} \
} while ( 0 )

static void bls_sc_packm_var
     (
             num_t      dt,
             dim_t      m,
             dim_t      k,
             dim_t      k_pack,
             dim_t      panel_dim_max,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t     coeff[BLS_SC_NQUADS],
             void*      p,
             inc_t      ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	const dim_t dt_size = bli_dt_size( dt );
	const dim_t n_iter  =
	    m / panel_dim_max + ( m % panel_dim_max != 0 );
	const dim_t nt      = bli_thrinfo_num_threads( thread );
	const dim_t tid     = bli_thrinfo_thread_id( thread );

	dim_t  source[BLS_SC_NQUADS];
	dim_t  n_source = 0;
	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
		if ( coeff[q] != 0 ) source[n_source++] = q;
	const bool simd_2src =
	    n_source == 2 &&
	    bls_sc_simd_pack_supported
	    ( dt, panel_dim_max, rs_src, cs_src );

	packm_cxk_ker_ft packm_ker =
	    bli_cntx_get_ukr2_dt( dt, dt, BLIS_PACKM_KER, cntx );
	const void* one  = bli_obj_buffer_for_1x1( dt, &BLIS_ONE );
	const void* mone = bli_obj_buffer_for_1x1( dt, &BLIS_MINUS_ONE );

	dim_t it_start, it_end, it_inc;
	bli_thread_range_slrr
	(
	  tid, nt, n_iter, 1, FALSE,
	  &it_start, &it_end, &it_inc
	);
	( void )it_inc;

	for ( dim_t it = 0; it < n_iter; ++it )
	{
		if ( !bli_is_my_iter( it, it_start, it_end, tid, nt ) ) continue;

		const dim_t row0     = it * panel_dim_max;
		const dim_t panel_m = bli_min( panel_dim_max, m - row0 );
		char*       p_panel  = ( char* )p + it * ps_p * dt_size;

		/*
		 * Single-source forms can use the architecture's native packing
		 * kernel, including its optimized transpose/interleave path.
		 */
		if ( n_source == 1 && packm_ker != NULL )
		{
			const dim_t q = source[0];
			const void* x = bls_sc_const_ptr_offset
			(
			  src[q], row0, 0, rs_src, cs_src, dt_size
			);

			packm_ker
			(
			  BLIS_NO_CONJUGATE,
			  BLIS_PACKED_PANELS,
			  panel_m,
			  panel_dim_max,
			  1,
			  k,
			  k_pack,
			  coeff[q] > 0 ? one : mone,
			  x, rs_src, cs_src,
			  p_panel, panel_dim_max,
			  NULL,
			  cntx
			);

			const inc_t panel_used = panel_dim_max * k_pack;
			if ( panel_used < ps_p )
				memset
				(
				  p_panel + panel_used * dt_size,
				  0,
				  ( ps_p - panel_used ) * dt_size
				);
			continue;
		}

		/*
		 * The loops below write the logical K region, including short-edge
		 * row padding. Clear only the long-edge K padding and the optional
		 * element used to make an odd canonical panel stride even.
		 */
		const inc_t panel_used = panel_dim_max * k;
		if ( panel_used < ps_p )
			memset
			(
			  p_panel + panel_used * dt_size,
			  0,
			  ( ps_p - panel_used ) * dt_size
			);

		if ( simd_2src )
		{
			const dim_t q0 = source[0];
			const dim_t q1 = source[1];
			const void* x0 = bls_sc_const_ptr_offset
			(
			  src[q0], row0, 0, rs_src, cs_src, dt_size
			);
			const void* x1 = bls_sc_const_ptr_offset
			(
			  src[q1], row0, 0, rs_src, cs_src, dt_size
			);

			if ( bls_sc_packm_2src_armv8a
			     (
			       dt, panel_m, k, panel_dim_max,
			       x0, x1, rs_src, cs_src,
			       coeff[q0], coeff[q1], p_panel
			     ) )
				continue;
		}

		if ( dt == BLIS_FLOAT )
		{
			float* restrict dst = ( float* )p_panel;

			if ( n_source == 2 )
			{
				const dim_t q0 = source[0];
				const dim_t q1 = source[1];
				const float* restrict x0 =
				    ( const float* )src[q0] + row0 * rs_src;
				const float* restrict x1 =
				    ( const float* )src[q1] + row0 * rs_src;

				if ( coeff[q0] > 0 )
				{
					if ( coeff[q1] > 0 )
						BLS_SC_PACK_PANEL( x0[off] + x1[off] );
					else
						BLS_SC_PACK_PANEL( x0[off] - x1[off] );
				}
				else
				{
					if ( coeff[q1] > 0 )
						BLS_SC_PACK_PANEL( x1[off] - x0[off] );
					else
						BLS_SC_PACK_PANEL( -x0[off] - x1[off] );
				}
			}
			else
			{
				for ( dim_t l = 0; l < k; ++l )
				for ( dim_t i = 0; i < panel_dim_max; ++i )
				{
					float v = 0.0F;

					if ( i < panel_m )
						for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
							v += ( float )coeff[q] *
							     ( ( const float* )src[q] )
							     [( row0 + i ) * rs_src +
							       l * cs_src];

					dst[l * panel_dim_max + i] = v;
				}
			}
		}
		else
		{
			double* restrict dst = ( double* )p_panel;

			if ( n_source == 2 )
			{
				const dim_t q0 = source[0];
				const dim_t q1 = source[1];
				const double* restrict x0 =
				    ( const double* )src[q0] + row0 * rs_src;
				const double* restrict x1 =
				    ( const double* )src[q1] + row0 * rs_src;

				if ( coeff[q0] > 0 )
				{
					if ( coeff[q1] > 0 )
						BLS_SC_PACK_PANEL( x0[off] + x1[off] );
					else
						BLS_SC_PACK_PANEL( x0[off] - x1[off] );
				}
				else
				{
					if ( coeff[q1] > 0 )
						BLS_SC_PACK_PANEL( x1[off] - x0[off] );
					else
						BLS_SC_PACK_PANEL( -x0[off] - x1[off] );
				}
			}
			else
			{
				for ( dim_t l = 0; l < k; ++l )
				for ( dim_t i = 0; i < panel_dim_max; ++i )
				{
					double v = 0.0;

					if ( i < panel_m )
						for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
							v += ( double )coeff[q] *
							     ( ( const double* )src[q] )
							     [( row0 + i ) * rs_src +
							       l * cs_src];

					dst[l * panel_dim_max + i] = v;
				}
			}
		}
	}
}

#undef BLS_SC_PACK_PANEL

static bool bls_sc_pair_plan_init
     (
       const int8_t coeff0[BLS_SC_NQUADS],
       const int8_t coeff1[BLS_SC_NQUADS],
             bls_sc_pair_plan_t* plan
     )
{
	plan->shared = BLS_SC_NQUADS;
	for ( dim_t form = 0; form < BLS_SC_BATCH_CAPACITY; ++form )
	{
		plan->unique[form] = BLS_SC_NQUADS;
		plan->shared_coeff[form] = 0;
		plan->unique_coeff[form] = 0;
	}

	dim_t n_shared = 0;
	dim_t n_source[BLS_SC_BATCH_CAPACITY] = { 0, 0 };

	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
	{
		const int8_t c0 = coeff0[q];
		const int8_t c1 = coeff1[q];
		if ( ( c0 < -1 || c0 > 1 ) || ( c1 < -1 || c1 > 1 ) )
			return FALSE;

		if ( c0 != 0 ) n_source[0] += 1;
		if ( c1 != 0 ) n_source[1] += 1;
		if ( c0 != 0 && c1 != 0 )
		{
			plan->shared = q;
			plan->shared_coeff[0] = c0;
			plan->shared_coeff[1] = c1;
			n_shared += 1;
		}
	}

	if ( n_shared != 1 ||
	     n_source[0] < 1 || n_source[0] > 2 ||
	     n_source[1] < 1 || n_source[1] > 2 )
		return FALSE;

	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
	{
		if ( q == plan->shared ) continue;
		if ( coeff0[q] != 0 )
		{
			if ( plan->unique[0] != BLS_SC_NQUADS ) return FALSE;
			plan->unique[0] = q;
			plan->unique_coeff[0] = coeff0[q];
		}
		if ( coeff1[q] != 0 )
		{
			if ( plan->unique[1] != BLS_SC_NQUADS ) return FALSE;
			plan->unique[1] = q;
			plan->unique_coeff[1] = coeff1[q];
		}
	}

	return TRUE;
}

/*
 * Match the ordinary and armv8a pair packers' canonical expression order.
 * In particular, (-shared) + unique is evaluated as unique - shared rather
 * than as a negation followed by an addition. Besides keeping the scalar
 * control bitwise comparable for finite inputs, this also preserves the
 * selected subtraction's signed-zero and exceptional-value behavior.
 */
static inline float bls_sc_pair_form_scalar_s
     (
             float   shared,
       const float*  unique,
             inc_t   off,
             int8_t shared_sign,
             int8_t unique_sign
     )
{
	if ( unique == NULL ) return shared_sign > 0 ? shared : -shared;

	const float other = unique[off];

	if ( shared_sign > 0 )
		return unique_sign > 0 ? shared + other : shared - other;

	return unique_sign > 0 ? other - shared : -shared - other;
}

static inline double bls_sc_pair_form_scalar_d
     (
             double   shared,
       const double*  unique,
             inc_t    off,
             int8_t  shared_sign,
             int8_t  unique_sign
     )
{
	if ( unique == NULL ) return shared_sign > 0 ? shared : -shared;

	const double other = unique[off];

	if ( shared_sign > 0 )
		return unique_sign > 0 ? shared + other : shared - other;

	return unique_sign > 0 ? other - shared : -shared - other;
}

static void bls_sc_packm_pair_var
     (
             num_t              dt,
             dim_t              m,
             dim_t              k,
             dim_t              panel_dim,
       const void*              src[BLS_SC_NQUADS],
             inc_t              rs_src,
             inc_t              cs_src,
       const bls_sc_pair_plan_t* plan,
             bool               simd_pack,
             void*              p0,
             void*              p1,
             inc_t              ps_p,
             thrinfo_t*         thread
     )
{
	const dim_t dt_size = bli_dt_size( dt );
	const dim_t n_iter =
	    m / panel_dim + ( m % panel_dim != 0 );
	const dim_t nt  = bli_thrinfo_num_threads( thread );
	const dim_t tid = bli_thrinfo_thread_id( thread );

	dim_t it_start, it_end, it_inc;
	bli_thread_range_slrr
	(
	  tid, nt, n_iter, 1, FALSE,
	  &it_start, &it_end, &it_inc
	);
	( void )it_inc;

	for ( dim_t it = 0; it < n_iter; ++it )
	{
		if ( !bli_is_my_iter( it, it_start, it_end, tid, nt ) ) continue;

		const dim_t row0 = it * panel_dim;
		const dim_t panel_m = bli_min( panel_dim, m - row0 );
		char* p0_panel = ( char* )p0 + it * ps_p * dt_size;
		char* p1_panel = ( char* )p1 + it * ps_p * dt_size;
		const inc_t panel_used = panel_dim * k;

		if ( panel_used < ps_p )
		{
			const siz_t padding = ( ps_p - panel_used ) * dt_size;
			memset( p0_panel + panel_used * dt_size, 0, padding );
			memset( p1_panel + panel_used * dt_size, 0, padding );
		}

		if ( simd_pack )
		{
			const void* shared = bls_sc_const_ptr_offset
			(
			  src[plan->shared], row0, 0, rs_src, cs_src, dt_size
			);
			const void* unique[BLS_SC_BATCH_CAPACITY] = { NULL, NULL };
			void* packed[BLS_SC_BATCH_CAPACITY] = { p0_panel, p1_panel };

			for ( dim_t form = 0;
			      form < BLS_SC_BATCH_CAPACITY;
			      ++form )
				if ( plan->unique[form] < BLS_SC_NQUADS )
					unique[form] = bls_sc_const_ptr_offset
					(
					  src[plan->unique[form]], row0, 0,
					  rs_src, cs_src, dt_size
					);

			if ( bls_sc_packm_pair_armv8a
			     (
			       dt, panel_m, k, panel_dim,
			       shared, unique, rs_src, cs_src,
			       plan->shared_coeff, plan->unique_coeff, packed
			     ) )
				continue;

			bli_print_msg
			(
			  "BLIS SubCuber SIMD pair-pack eligibility mismatch",
			  __FILE__, __LINE__
			);
			bli_abort();
		}

		if ( dt == BLIS_FLOAT )
		{
			float* restrict dst0 = ( float* )p0_panel;
			float* restrict dst1 = ( float* )p1_panel;
			const float* shared = ( const float* )src[plan->shared];
			const float* unique0 =
			    plan->unique[0] < BLS_SC_NQUADS
			      ? ( const float* )src[plan->unique[0]]
			      : NULL;
			const float* unique1 =
			    plan->unique[1] < BLS_SC_NQUADS
			      ? ( const float* )src[plan->unique[1]]
			      : NULL;

			for ( dim_t l = 0; l < k; ++l )
			{
				dim_t i = 0;
				for ( ; i < panel_m; ++i )
				{
					const inc_t off =
					    ( row0 + i ) * rs_src + l * cs_src;
					const float xs = shared[off];
					const float y0 = bls_sc_pair_form_scalar_s
					(
					  xs, unique0, off,
					  plan->shared_coeff[0], plan->unique_coeff[0]
					);
					const float y1 = bls_sc_pair_form_scalar_s
					(
					  xs, unique1, off,
					  plan->shared_coeff[1], plan->unique_coeff[1]
					);

					dst0[l * panel_dim + i] = y0;
					dst1[l * panel_dim + i] = y1;
				}
				for ( ; i < panel_dim; ++i )
				{
					dst0[l * panel_dim + i] = 0.0F;
					dst1[l * panel_dim + i] = 0.0F;
				}
			}
		}
		else
		{
			double* restrict dst0 = ( double* )p0_panel;
			double* restrict dst1 = ( double* )p1_panel;
			const double* shared = ( const double* )src[plan->shared];
			const double* unique0 =
			    plan->unique[0] < BLS_SC_NQUADS
			      ? ( const double* )src[plan->unique[0]]
			      : NULL;
			const double* unique1 =
			    plan->unique[1] < BLS_SC_NQUADS
			      ? ( const double* )src[plan->unique[1]]
			      : NULL;

			for ( dim_t l = 0; l < k; ++l )
			{
				dim_t i = 0;
				for ( ; i < panel_m; ++i )
				{
					const inc_t off =
					    ( row0 + i ) * rs_src + l * cs_src;
					const double xs = shared[off];
					const double y0 = bls_sc_pair_form_scalar_d
					(
					  xs, unique0, off,
					  plan->shared_coeff[0], plan->unique_coeff[0]
					);
					const double y1 = bls_sc_pair_form_scalar_d
					(
					  xs, unique1, off,
					  plan->shared_coeff[1], plan->unique_coeff[1]
					);

					dst0[l * panel_dim + i] = y0;
					dst1[l * panel_dim + i] = y1;
				}
				for ( ; i < panel_dim; ++i )
				{
					dst0[l * panel_dim + i] = 0.0;
					dst1[l * panel_dim + i] = 0.0;
				}
			}
		}
	}
}

static void bls_sc_packm_batch
     (
             num_t      dt,
             dim_t      m_alloc,
             dim_t      k_alloc,
             dim_t      m,
             dim_t      k,
             dim_t      panel_dim,
             dim_t      k_align,
             packbuf_t  pack_buf_type,
             dim_t      generation,
             dim_t      form_capacity,
             dim_t      n_forms,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t* const coeff[BLS_SC_BATCH_CAPACITY],
             void*      p[BLS_SC_BATCH_CAPACITY],
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	/*
	 * Reserve aligned slots sized for the largest block this communicator
	 * will see and for a fixed number of forms per batch. Fixing capacity at
	 * the first singleton batch prevents a later pair from resizing storage
	 * while peers consume the previous generation. A single-thread
	 * communicator cannot overlap packing with a peer, so it uses one slot;
	 * shared communicators use two.
	 *
	 * Double-buffer safety depends on a precise collective invariant:
	 * every member of this buffer-owning communicator calls pack in the
	 * same generation order. The terminal barrier below first makes all
	 * packed panels ready for consumers and then limits skew to one
	 * generation. Thus a fast thread may write generation g+1 while a peer
	 * consumes g, but it cannot reach g+2 (which reuses g's slot) until
	 * that peer joins the g+1 pack collective. Keep the barrier on exactly
	 * this thrinfo communicator and keep generation parity coupled to the
	 * pack call sequence.
	 */
	const dim_t dim_max = ( dim_t )( ( ( siz_t )-1 ) >> 1 );

	if ( m_alloc < 0 || k_alloc < 0 || m < 0 || k < 0 ||
	     m > m_alloc || k > k_alloc ||
	     panel_dim <= 0 || k_align <= 0 ||
	     form_capacity < 1 ||
	     form_capacity > BLS_SC_BATCH_CAPACITY ||
	     n_forms < 1 || n_forms > form_capacity ||
	     k > dim_max - ( k_align - 1 ) ||
	     k_alloc > dim_max - ( k_align - 1 ) )
		bls_sc_pack_size_abort( thread );

	const dim_t n_panels_alloc =
	    m_alloc / panel_dim + ( m_alloc % panel_dim != 0 );
	const dim_t k_pack =
	    bli_align_dim_to_mult( k, k_align, TRUE );
	const dim_t k_pack_alloc =
	    bli_align_dim_to_mult( k_alloc, k_align, TRUE );
	siz_t panel_stride_size = 0;
	siz_t panel_stride_alloc_size = 0;

	if ( !bls_sc_size_mul
	      ( ( siz_t )panel_dim, ( siz_t )k_pack,
	        &panel_stride_size ) ||
	     !bls_sc_size_mul
	      ( ( siz_t )panel_dim, ( siz_t )k_pack_alloc,
	        &panel_stride_alloc_size ) ||
	     panel_stride_size > ( siz_t )dim_max - 1 ||
	     panel_stride_alloc_size > ( siz_t )dim_max - 1 )
		bls_sc_pack_size_abort( thread );

	inc_t panel_stride = ( inc_t )panel_stride_size;
	inc_t panel_stride_alloc = ( inc_t )panel_stride_alloc_size;

	if ( bli_is_odd( panel_stride ) ) panel_stride += 1;
	if ( bli_is_odd( panel_stride_alloc ) ) panel_stride_alloc += 1;

	const siz_t pool_align_size =
	    pack_buf_type == BLIS_BUFFER_FOR_A_BLOCK
	      ? BLIS_POOL_ADDR_ALIGN_SIZE_A
	      : BLIS_POOL_ADDR_ALIGN_SIZE_B;
	const dim_t n_slots =
	    bli_thrinfo_num_threads( thread ) > 1 ? 2 : 1;
	siz_t form_elements = 0;
	siz_t form_size_raw = 0;
	siz_t form_size = 0;
	siz_t batch_slot_size = 0;

	if ( !bls_sc_size_mul
	      ( ( siz_t )n_panels_alloc,
	        ( siz_t )panel_stride_alloc, &form_elements ) ||
	     !bls_sc_size_mul
	      ( ( siz_t )bli_dt_size( dt ),
	        form_elements, &form_size_raw ) ||
	     !bls_sc_size_round_up
	      ( form_size_raw, pool_align_size, &form_size ) ||
	     !bls_sc_size_mul
	      ( ( siz_t )form_capacity, form_size, &batch_slot_size ) )
		bls_sc_pack_size_abort( thread );

	/*
	 * Preserve the A/B buffer type even when the full request exceeds the
	 * architecture's initial block size. BLIS grows that pool under the pba
	 * lock and retains returned blocks across GEMMs. Sending oversized
	 * requests to GEN_USE instead would malloc/free them with every
	 * short-lived SUP tree and would also weaken the A/B alignment contract.
	 */
	siz_t alloc_size;

	if ( !bls_sc_size_mul
	      ( ( siz_t )n_slots, batch_slot_size, &alloc_size ) )
		bls_sc_pack_size_abort( thread );

#ifndef BLIS_ENABLE_PBA_POOLS
	/*
	 * With pools disabled, BLIS internally maps A/B requests to GEN_USE.
	 * Reserve enough slack to restore the selected pack pool's alignment
	 * and address offset below.
	 */
	if ( !bls_sc_size_add
	      ( alloc_size, pool_align_size - 1, &alloc_size ) )
		bls_sc_pack_size_abort( thread );
#endif

#ifdef BLIS_SUBCUBER_DEBUG_PACK_GENERATIONS
	/*
	 * Opt-in race guard for synchronization changes: reserve an aligned
	 * per-slot, per-participant stamp table. Before packing generation g, a
	 * thread verifies that every participant finished packing g-1. It marks
	 * its own portion of g complete only after its panel work. Thus an early
	 * non-chief writer is detected before it can reuse a slow consumer's
	 * slot, and a missing readiness barrier is detected before computation.
	 */
	siz_t generation_header_size = 0;
	if ( n_slots == 2 )
	{
		siz_t stamp_count;
		siz_t stamp_bytes;

		if ( !bls_sc_size_mul
		      ( 2, ( siz_t )bli_thrinfo_num_threads( thread ),
		        &stamp_count ) ||
		     !bls_sc_size_mul
		      ( stamp_count, sizeof( dim_t ), &stamp_bytes ) ||
		     !bls_sc_size_round_up
		      ( stamp_bytes, pool_align_size,
		        &generation_header_size ) ||
		     !bls_sc_size_add
		      ( alloc_size, generation_header_size, &alloc_size ) )
			bls_sc_pack_size_abort( thread );
	}
#endif

	void* const p_base =
	    bli_packm_alloc_ex( alloc_size, pack_buf_type, thread );

	char* p_aligned = p_base;

#ifndef BLIS_ENABLE_PBA_POOLS
	const siz_t pool_offset_size =
	    pack_buf_type == BLIS_BUFFER_FOR_A_BLOCK
	      ? BLIS_POOL_ADDR_OFFSET_SIZE_A
	      : BLIS_POOL_ADDR_OFFSET_SIZE_B;
	const uintptr_t address = ( uintptr_t )p_aligned;
	const siz_t current_residue = address % pool_align_size;
	const siz_t wanted_residue = pool_offset_size % pool_align_size;
	const siz_t alignment_delta =
	    ( wanted_residue + pool_align_size - current_residue ) %
	    pool_align_size;
	p_aligned += alignment_delta;
#endif

	char* p_slots = p_aligned;

#ifdef BLIS_SUBCUBER_DEBUG_PACK_GENERATIONS
	dim_t* generation_stamps = NULL;
	dim_t  generation_nt = 0;
	dim_t  generation_tid = 0;

	if ( n_slots == 2 )
	{
		generation_stamps = ( dim_t* )p_aligned;
		generation_nt = bli_thrinfo_num_threads( thread );
		generation_tid = bli_thrinfo_thread_id( thread );
		p_slots += generation_header_size;

		if ( generation > 0 )
		{
			const dim_t previous = ( generation - 1 ) & 1;
			for ( dim_t participant = 0;
			      participant < generation_nt;
			      ++participant )
			{
				const dim_t observed =
				    __atomic_load_n
				    (
				      &generation_stamps
				          [previous * generation_nt + participant],
				      __ATOMIC_ACQUIRE
				    );
				if ( observed != generation - 1 )
				{
					bli_print_msg
					(
					  "BLIS SubCuber previous pack generation "
					  "is incomplete",
					  __FILE__,
					  __LINE__
					);
					bli_abort();
				}
			}
		}

		if ( generation_tid < 0 || generation_tid >= generation_nt )
		{
			bli_print_msg
			(
			  "BLIS SubCuber pack-buffer participant id invalid",
			  __FILE__,
			  __LINE__
			);
			bli_abort();
		}
	}
#endif

	const dim_t slot = n_slots == 2 ? ( generation & 1 ) : 0;

	*rs_p = 1;
	*cs_p = panel_dim;
	*ps_p = panel_stride;

	for ( dim_t form = 0; form < n_forms; ++form )
		p[form] = p_slots +
		          slot * batch_slot_size +
		          form * form_size;

	bool packed_pair = FALSE;
	if ( n_forms == BLS_SC_BATCH_CAPACITY )
	{
		bls_sc_pair_plan_t plan;
		if ( bls_sc_pair_plan_init( coeff[0], coeff[1], &plan ) )
		{
			const bool simd_pair = bls_sc_pair_simd_pack_supported
			(
			  dt, m, panel_dim, rs_src, cs_src
			);
			const bool scalar_pair = bls_sc_env_flag
			(
			  "BLIS_SUBCUBER_EXPERIMENT_PAIR_FUSED_PACK"
			);

			if ( simd_pair || scalar_pair )
			{
				bls_sc_packm_pair_var
				(
				  dt, m, k, panel_dim,
				  src, rs_src, cs_src, &plan, simd_pair,
				  p[0], p[1], *ps_p, thread
				);
				packed_pair = TRUE;
			}
		}
	}

	if ( !packed_pair )
		for ( dim_t form = 0; form < n_forms; ++form )
			bls_sc_packm_var
			(
			  dt, m, k, k_pack, panel_dim,
			  src, rs_src, cs_src, coeff[form],
			  p[form], *ps_p, cntx, thread
			);

#ifdef BLIS_SUBCUBER_DEBUG_PACK_GENERATIONS
	if ( n_slots == 2 )
		__atomic_store_n
		(
		  &generation_stamps
		      [( generation & 1 ) * generation_nt + generation_tid],
		  generation,
		  __ATOMIC_RELEASE
		);
#endif

	bli_thrinfo_barrier( thread );

#ifdef BLIS_SUBCUBER_DEBUG_PACK_GENERATIONS
	/*
	 * The terminal barrier is also the pack-readiness barrier: no consumer
	 * may return until every participant has finished its panels. Checking
	 * completion after it makes a removed or weakened barrier fail before
	 * the microkernel can observe a partially packed operand.
	 */
	if ( n_slots == 2 )
	{
		const dim_t current = generation & 1;
		for ( dim_t participant = 0;
		      participant < generation_nt;
		      ++participant )
		{
			const dim_t observed =
			    __atomic_load_n
			    (
			      &generation_stamps
			          [current * generation_nt + participant],
			      __ATOMIC_ACQUIRE
			    );
			if ( observed != generation )
			{
				bli_print_msg
				(
				  "BLIS SubCuber current pack generation "
				  "is incomplete after its barrier",
				  __FILE__,
				  __LINE__
				);
				bli_abort();
			}
		}
	}
#endif
}

static void bls_sc_packm
     (
             num_t      dt,
             dim_t      m_alloc,
             dim_t      k_alloc,
             dim_t      m,
             dim_t      k,
             dim_t      panel_dim,
             dim_t      k_align,
             packbuf_t  pack_buf_type,
             dim_t      generation,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t     coeff[BLS_SC_NQUADS],
             void**     p,
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	const int8_t* coeff_batch[BLS_SC_BATCH_CAPACITY] =
	    { coeff, NULL };
	void* packed[BLS_SC_BATCH_CAPACITY] = { NULL, NULL };

	bls_sc_packm_batch
	(
	  dt, m_alloc, k_alloc, m, k, panel_dim, k_align,
	  pack_buf_type, generation, 1, 1,
	  src, rs_src, cs_src, coeff_batch,
	  packed, rs_p, cs_p, ps_p, cntx, thread
	);

	*p = packed[0];
}

static void bls_sc_packm_a
     (
             num_t      dt,
             dim_t      m_alloc,
             dim_t      k_alloc,
             dim_t      m,
             dim_t      k,
             dim_t      mr,
             dim_t      kr,
             dim_t      generation,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t     coeff[BLS_SC_NQUADS],
             void**     p,
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	bls_sc_packm
	(
	  dt, m_alloc, k_alloc, m, k, mr, kr,
	  BLIS_BUFFER_FOR_A_BLOCK, generation,
	  src, rs_src, cs_src, coeff,
	  p, rs_p, cs_p, ps_p, cntx, thread
	);
}

static void bls_sc_packm_b
     (
             num_t      dt,
             dim_t      k_alloc,
             dim_t      n_alloc,
             dim_t      k,
             dim_t      n,
             dim_t      nr,
             dim_t      kr,
             dim_t      generation,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t     coeff[BLS_SC_NQUADS],
             void**     p,
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	/*
	 * Pack B as the implicit transpose used by the conventional BLIS
	 * block-panel algorithm: the logical panel rows are output columns.
	 */
	bls_sc_packm
	(
	  dt, n_alloc, k_alloc, n, k, nr, kr,
	  BLIS_BUFFER_FOR_B_PANEL, generation,
	  src, cs_src, rs_src, coeff,
	  p, cs_p, rs_p, ps_p, cntx, thread
	);
}

static void bls_sc_packm_a_batch
     (
             num_t      dt,
             dim_t      m_alloc,
             dim_t      k_alloc,
             dim_t      m,
             dim_t      k,
             dim_t      mr,
             dim_t      kr,
             dim_t      generation,
             dim_t      n_forms,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t* const coeff[BLS_SC_BATCH_CAPACITY],
             void*      p[BLS_SC_BATCH_CAPACITY],
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	bls_sc_packm_batch
	(
	  dt, m_alloc, k_alloc, m, k, mr, kr,
	  BLIS_BUFFER_FOR_A_BLOCK, generation,
	  BLS_SC_BATCH_CAPACITY, n_forms,
	  src, rs_src, cs_src, coeff,
	  p, rs_p, cs_p, ps_p, cntx, thread
	);
}

static void bls_sc_packm_b_batch
     (
             num_t      dt,
             dim_t      k_alloc,
             dim_t      n_alloc,
             dim_t      k,
             dim_t      n,
             dim_t      nr,
             dim_t      kr,
             dim_t      generation,
             dim_t      n_forms,
       const void*      src[BLS_SC_NQUADS],
             inc_t      rs_src,
             inc_t      cs_src,
       const int8_t* const coeff[BLS_SC_BATCH_CAPACITY],
             void*      p[BLS_SC_BATCH_CAPACITY],
             inc_t*     rs_p,
             inc_t*     cs_p,
             inc_t*     ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	bls_sc_packm_batch
	(
	  dt, n_alloc, k_alloc, n, k, nr, kr,
	  BLIS_BUFFER_FOR_B_PANEL, generation,
	  BLS_SC_BATCH_CAPACITY, n_forms,
	  src, cs_src, rs_src, coeff,
	  p, cs_p, rs_p, ps_p, cntx, thread
	);
}

#define BLS_SC_SCATTER_ROW( stmt ) \
do \
{ \
	for ( dim_t i = 0; i < m; ++i ) \
	for ( dim_t j = 0; j < n; ++j ) \
	{ \
		const inc_t ci = i * rs_c + j * cs_c; \
		const inc_t ti = i * rs_t + j * cs_t; \
		stmt; \
	} \
} while ( 0 )

#define BLS_SC_SCATTER_COL( stmt ) \
do \
{ \
	for ( dim_t j = 0; j < n; ++j ) \
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		const inc_t ci = i * rs_c + j * cs_c; \
		const inc_t ti = i * rs_t + j * cs_t; \
		stmt; \
	} \
} while ( 0 )

#define BLS_SC_SCATTER( stmt ) \
do \
{ \
	if ( bli_abs( cs_c ) <= bli_abs( rs_c ) ) \
		BLS_SC_SCATTER_ROW( stmt ); \
	else \
		BLS_SC_SCATTER_COL( stmt ); \
} while ( 0 )

#define BLS_SC_UPDATE_DST( cq, value, sign, init, beta_value ) \
do \
{ \
	if ( init ) \
	{ \
		if ( beta_value == 0 ) \
			cq[ci] = sign > 0 ? value : -value; \
		else \
			cq[ci] = beta_value * cq[ci] + \
			         ( sign > 0 ? value : -value ); \
	} \
	else if ( sign > 0 ) cq[ci] += value; \
	else                 cq[ci] -= value; \
} while ( 0 )

static void bls_sc_scatter_tile
     (
             num_t      dt,
             dim_t      m,
             dim_t      n,
       const void*      t,
             inc_t      rs_t,
             inc_t      cs_t,
             void*      c_quads[BLS_SC_NQUADS],
             inc_t      rs_c,
             inc_t      cs_c,
       const int8_t     coeff[BLS_SC_NQUADS],
       const void*      beta,
             uint8_t    init_mask
     )
{
	dim_t dest[2] = { BLS_SC_NQUADS, BLS_SC_NQUADS };
	dim_t n_dest  = 0;

	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
		if ( coeff[q] != 0 )
		{
			if ( n_dest < 2 ) dest[n_dest] = q;
			n_dest += 1;
		}

	/*
	 * The selected tensor guarantees one or two destinations. Resolve them
	 * before entering the element loop so a two-way fanout loads each
	 * scratch value only once.
	 */
	if ( n_dest == 0 || n_dest > 2 ) return;

	const dim_t  q0    = dest[0];
	const dim_t  q1    = dest[1];
	const int8_t sign0 = coeff[q0];
	const int8_t sign1 = q1 < BLS_SC_NQUADS ? coeff[q1] : 0;
	const bool   init0 = init_mask & ( ( uint8_t )1 << q0 );
	const bool   init1 =
	    q1 < BLS_SC_NQUADS &&
	    ( init_mask & ( ( uint8_t )1 << q1 ) );

	if ( dt == BLIS_FLOAT )
	{
		const float* tv = ( const float* )t;
		const float  bv = *( const float* )beta;
		float* cq0 = ( float* )c_quads[q0];

		if ( q1 < BLS_SC_NQUADS )
		{
			float* cq1 = ( float* )c_quads[q1];
			BLS_SC_SCATTER
			(
			  const float v = tv[ti];
			  BLS_SC_UPDATE_DST( cq0, v, sign0, init0, bv );
			  BLS_SC_UPDATE_DST( cq1, v, sign1, init1, bv )
			);
		}
		else
			BLS_SC_SCATTER
			(
			  const float v = tv[ti];
			  BLS_SC_UPDATE_DST( cq0, v, sign0, init0, bv )
			);
	}
	else
	{
		const double* tv = ( const double* )t;
		const double  bv = *( const double* )beta;
		double* cq0 = ( double* )c_quads[q0];

		if ( q1 < BLS_SC_NQUADS )
		{
			double* cq1 = ( double* )c_quads[q1];
			BLS_SC_SCATTER
			(
			  const double v = tv[ti];
			  BLS_SC_UPDATE_DST( cq0, v, sign0, init0, bv );
			  BLS_SC_UPDATE_DST( cq1, v, sign1, init1, bv )
			);
		}
		else
			BLS_SC_SCATTER
			(
			  const double v = tv[ti];
			  BLS_SC_UPDATE_DST( cq0, v, sign0, init0, bv )
			);
	}
}

static void bls_sc_fanout_c_tile
     (
             num_t      dt,
             dim_t      m,
             dim_t      n,
       const void*      t,
             inc_t      rs_t,
             inc_t      cs_t,
             void*      c,
             inc_t      rs_c,
             inc_t      cs_c,
             int8_t     sign
     )
{
	/*
	 * A product that seeds a quadrant with beta == zero may use that C tile
	 * as its temporary. Fan the just-written product into its second
	 * destination before later terms modify the seed tile.
	 */
	if ( dt == BLIS_FLOAT )
	{
		const float* tv = ( const float* )t;
		      float* cv = (       float* )c;

		if ( sign > 0 )
			BLS_SC_SCATTER( cv[ci] += tv[ti] );
		else
			BLS_SC_SCATTER( cv[ci] -= tv[ti] );
	}
	else
	{
		const double* tv = ( const double* )t;
		      double* cv = (       double* )c;

		if ( sign > 0 )
			BLS_SC_SCATTER( cv[ci] += tv[ti] );
		else
			BLS_SC_SCATTER( cv[ci] -= tv[ti] );
	}
}

#undef BLS_SC_UPDATE_DST
#undef BLS_SC_SCATTER
#undef BLS_SC_SCATTER_COL
#undef BLS_SC_SCATTER_ROW

static bls_sc_term_exec_t bls_sc_prepare_term
     (
             dim_t  term,
             dim_t  pp,
       const dim_t  init_term[BLS_SC_NQUADS],
             bool   direct_c,
             bool   beta_zero
     )
{
	bls_sc_term_exec_t exec =
	{
		.term = term,
		.init_mask = 0,
		.direct_q = BLS_SC_NQUADS,
		.fanout_q = BLS_SC_NQUADS,
		.two_q0 = BLS_SC_NQUADS,
		.two_q1 = BLS_SC_NQUADS,
	};

	if ( pp == 0 )
		for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
			if ( init_term[q] == term )
				exec.init_mask |= ( uint8_t )1 << q;

	dim_t n_dest = 0;
	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
		if ( bls_sc_terms[term].c[q] != 0 )
		{
			n_dest += 1;
			if ( bls_sc_terms[term].c[q] == 1 &&
			     exec.two_q0 == BLS_SC_NQUADS )
				exec.two_q0 = q;
		}

	if ( n_dest == 2 && exec.two_q0 < BLS_SC_NQUADS )
		for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
			if ( q != exec.two_q0 && bls_sc_terms[term].c[q] != 0 )
			{
				exec.two_q1 = q;
				break;
			}

	if ( !direct_c ) return exec;

	for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
		if ( bls_sc_terms[term].c[q] != 0 ) exec.direct_q = q;

	if ( n_dest != 1 || bls_sc_terms[term].c[exec.direct_q] != 1 )
		exec.direct_q = BLS_SC_NQUADS;

	/*
	 * With beta == zero, a positive first contribution is exactly M. Let
	 * the native kernel write M to that quadrant and fan it out from there.
	 */
	if ( exec.direct_q == BLS_SC_NQUADS && n_dest == 2 && beta_zero )
	{
		for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
			if ( ( exec.init_mask & ( ( uint8_t )1 << q ) ) &&
			     bls_sc_terms[term].c[q] == 1 )
			{
				exec.direct_q = q;
				break;
			}

		if ( exec.direct_q < BLS_SC_NQUADS )
			for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
				if ( q != exec.direct_q &&
				     bls_sc_terms[term].c[q] != 0 )
				{
					exec.fanout_q = q;
					break;
				}
	}

	return exec;
}

static void bls_sc_run_term_tile
     (
       const bls_sc_ukr_ctx_t* ctx,
       const bls_sc_term_exec_t* exec,
             dim_t              kc_cur,
             dim_t              mr_cur,
             dim_t              nr_cur,
       const void*              a_ir,
       const void*              b_jr,
       const void*              a2,
       const void*              b2,
             void*              c_tile[BLS_SC_NQUADS]
     )
{
	const dim_t term = exec->term;
	auxinfo_t aux = { 0 };
	bli_auxinfo_set_schema_a( BLIS_PACKED_PANELS, &aux );
	bli_auxinfo_set_schema_b( BLIS_PACKED_PANELS, &aux );
	bli_auxinfo_set_is_a( 1, &aux );
	bli_auxinfo_set_is_b( 1, &aux );
	bli_auxinfo_set_next_a( a2, &aux );
	bli_auxinfo_set_next_b( b2, &aux );

	if ( ctx->gemm_ukr_2c != NULL &&
	     exec->two_q1 < BLS_SC_NQUADS &&
	     mr_cur == ctx->mr && nr_cur == ctx->nr )
	{
		const void* beta0 =
		    exec->init_mask & ( ( uint8_t )1 << exec->two_q0 )
		      ? ctx->beta
		      : ctx->one;
		const void* beta1 =
		    exec->init_mask & ( ( uint8_t )1 << exec->two_q1 )
		      ? ctx->beta
		      : ctx->one;
		bls_sc_2c_params_t params =
		{
			.c2 = c_tile[exec->two_q1],
			.beta2 = beta1,
			.rs_c2 = ctx->rs_c,
			.relative_sign = bls_sc_terms[term].c[exec->two_q1]
		};
		bli_auxinfo_set_params( &params, &aux );

		ctx->gemm_ukr_2c
		(
		  ctx->mr, ctx->nr, kc_cur,
		  ctx->alpha, a_ir, b_jr, beta0,
		  c_tile[exec->two_q0], ctx->rs_c, ctx->cs_c,
		  &aux, ctx->cntx
		);
	}
	else if ( exec->direct_q < BLS_SC_NQUADS )
	{
		const void* beta_use =
		    exec->init_mask & ( ( uint8_t )1 << exec->direct_q )
		      ? ctx->beta
		      : ctx->one;

		ctx->gemm_ukr
		(
		  ctx->mr, ctx->nr, kc_cur,
		  ctx->alpha, a_ir, b_jr, beta_use,
		  c_tile[exec->direct_q], ctx->rs_c, ctx->cs_c,
		  &aux, ctx->cntx
		);

		if ( exec->fanout_q < BLS_SC_NQUADS )
			bls_sc_fanout_c_tile
			(
			  ctx->dt, mr_cur, nr_cur,
			  c_tile[exec->direct_q], ctx->rs_c, ctx->cs_c,
			  c_tile[exec->fanout_q], ctx->rs_c, ctx->cs_c,
			  bls_sc_terms[term].c[exec->fanout_q]
			);
	}
	else
	{
		char tile[BLIS_STACK_BUF_MAX_SIZE]
		     __attribute__(( aligned(BLIS_STACK_BUF_ALIGN_SIZE) ));

		ctx->gemm_ukr
		(
		  ctx->mr, ctx->nr, kc_cur,
		  ctx->alpha, a_ir, b_jr, ctx->zero,
		  tile, ctx->rs_t, ctx->cs_t,
		  &aux, ctx->cntx
		);

		bls_sc_scatter_tile
		(
		  ctx->dt, mr_cur, nr_cur,
		  tile, ctx->rs_t, ctx->cs_t,
		  c_tile, ctx->rs_c, ctx->cs_c,
		  bls_sc_terms[term].c,
		  ctx->beta, exec->init_mask
		);
	}
}

static void bls_sc_gemm_bp
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     )
{
	const num_t dt      = bli_obj_dt( c );
	const dim_t dt_size = bli_dt_size( dt );

	const dim_t m  = bli_obj_length( c );
	const dim_t n  = bli_obj_width( c );
	const dim_t k  = bli_obj_width( a );
	const dim_t mh = m / 2;
	const dim_t nh = n / 2;
	const dim_t kh = k / 2;

	const char* a00 = bli_obj_buffer_at_off( a );
	const inc_t rs_a = bli_obj_row_stride( a );
	const inc_t cs_a = bli_obj_col_stride( a );

	const char* b00 = bli_obj_buffer_at_off( b );
	const inc_t rs_b = bli_obj_row_stride( b );
	const inc_t cs_b = bli_obj_col_stride( b );

	char* c00 = bli_obj_buffer_at_off( c );
	const inc_t rs_c = bli_obj_row_stride( c );
	const inc_t cs_c = bli_obj_col_stride( c );

	const char* alpha_buf = bli_obj_buffer_for_1x1( dt, alpha );
	const char* beta_buf  = bli_obj_buffer_for_1x1( dt, beta );
	const char* zero      = bli_obj_buffer_for_1x1( dt, &BLIS_ZERO );
	const char* one       = bli_obj_buffer_for_1x1( dt, &BLIS_ONE );
	const bool beta_zero =
	    dt == BLIS_FLOAT
	      ? *( const float* )beta_buf == 0.0F
	      : *( const double* )beta_buf == 0.0;

	gemm_ukr_ft gemm_ukr =
	    bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx );
	const bool armv8a_2c =
	    bls_sc_2c_armv8a_is_compatible( dt, gemm_ukr, cntx );
	const bool direct_c =
	    bli_cntx_prefers_storage_of( c, BLIS_GEMM_UKR, cntx );
	const bool can_use_2c =
	    armv8a_2c && direct_c && rs_c > 0 && cs_c == 1;

	const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t       NC = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
	dim_t       MC = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
	dim_t       KC = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
	dim_t       KR = bli_cntx_get_blksz_def_dt( dt, BLIS_KR, cntx );
	if ( KR == 0 ) KR = 1;

	/*
	 * The fused SGEMM schedule amortizes its pack barriers better with
	 * 50 MR panels per block on Firestorm. Keep this local to SubCuber and
	 * apply the environment override afterwards.
	 */
	if ( dt == BLIS_FLOAT && can_use_2c ) MC = 50 * MR;

	NC = bls_sc_env_blksz( "BLIS_SUBCUBER_NC", NC, NR );
	MC = bls_sc_env_blksz( "BLIS_SUBCUBER_MC", MC, MR );
	KC = bls_sc_env_blksz( "BLIS_SUBCUBER_KC", KC, KR );

	const bool row_pref =
	    bli_cntx_get_ukr_prefs_dt( dt, BLIS_GEMM_UKR_ROW_PREF, cntx );
	const inc_t rs_t = row_pref ? NR : 1;
	const inc_t cs_t = row_pref ? 1 : MR;

	gemm_ukr_ft gemm_ukr_2c = NULL;

	if ( can_use_2c )
		gemm_ukr_2c =
		    dt == BLIS_FLOAT
		      ? bls_sc_sgemm_armv8a_asm_12x8r_2c
		      : bls_sc_dgemm_armv8a_asm_8x6r_2c;

	const bls_sc_ukr_ctx_t ukr_ctx =
	{
		.dt = dt,
		.mr = MR,
		.nr = NR,
		.rs_c = rs_c,
		.cs_c = cs_c,
		.rs_t = rs_t,
		.cs_t = cs_t,
		.alpha = alpha_buf,
		.beta = beta_buf,
		.zero = zero,
		.one = one,
		.gemm_ukr = gemm_ukr,
		.gemm_ukr_2c = gemm_ukr_2c,
		.cntx = cntx,
	};

	const char* a_quads[BLS_SC_NQUADS] =
	{
		a00,
		bls_sc_const_ptr_offset( a00, 0,  kh, rs_a, cs_a, dt_size ),
		bls_sc_const_ptr_offset( a00, mh, 0,  rs_a, cs_a, dt_size ),
		bls_sc_const_ptr_offset( a00, mh, kh, rs_a, cs_a, dt_size )
	};
	const char* b_quads[BLS_SC_NQUADS] =
	{
		b00,
		bls_sc_const_ptr_offset( b00, 0,  nh, rs_b, cs_b, dt_size ),
		bls_sc_const_ptr_offset( b00, kh, 0,  rs_b, cs_b, dt_size ),
		bls_sc_const_ptr_offset( b00, kh, nh, rs_b, cs_b, dt_size )
	};
	char* c_quads[BLS_SC_NQUADS] =
	{
		c00,
		bls_sc_ptr_offset( c00, 0,  nh, rs_c, cs_c, dt_size ),
		bls_sc_ptr_offset( c00, mh, 0,  rs_c, cs_c, dt_size ),
		bls_sc_ptr_offset( c00, mh, nh, rs_c, cs_c, dt_size )
	};

	bls_sc_thrinfo_path_t path;
	bls_sc_thrinfo_path_require( thread, rntm, &path );
	thrinfo_t* const thread_jc = path.jc;
	thrinfo_t* const thread_pb = path.pb;
	thrinfo_t* const thread_ic = path.ic;
	thrinfo_t* const thread_pa = path.pa;
	thrinfo_t* const thread_jr = path.jr;
	thrinfo_t* const thread_ir = path.ir;
	const bool pair_batch =
	    bls_sc_env_flag( "BLIS_SUBCUBER_EXPERIMENT_PAIR_BATCH" );
	const bool locality_trail =
	    bls_sc_env_flag( "BLIS_SUBCUBER_EXPERIMENT_TERM_TRAIL" );
	const dim_t* const term_order =
	    locality_trail
	      ? bls_sc_term_order_trail
	      : bls_sc_term_order_default;
	const dim_t* const init_term =
	    locality_trail && !pair_batch
	      ? bls_sc_init_term_trail
	      : bls_sc_init_term_default;
	dim_t b_pack_generation = 0;
	dim_t a_pack_generation = 0;

	dim_t jc_start, jc_end;
	const dim_t jc_tid = bli_thrinfo_work_id( thread_jc );
	const dim_t jc_nt  = bli_thrinfo_n_way( thread_jc );
	bli_thread_range_sub
	(
	  jc_tid, jc_nt, nh, NR, FALSE,
	  &jc_start, &jc_end
	);
	const dim_t jc_left = ( jc_end - jc_start ) % NC;
	const dim_t nc_alloc = bli_min( NC, jc_end - jc_start );
	const dim_t kc_alloc = bli_min( KC, kh );

	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );
		const dim_t pc_left = kh % KC;

		for ( dim_t pp = 0; pp < kh; pp += KC )
		{
			const dim_t kc_cur = ( KC <= kh - pp ? KC : pc_left );
			const dim_t n_schedule =
			    pair_batch ? BLS_SC_NBATCHES : BLS_SC_NTERMS;

			for ( dim_t schedule = 0; schedule < n_schedule; ++schedule )
			{
				dim_t n_forms;
				dim_t term[BLS_SC_BATCH_CAPACITY];

				if ( pair_batch )
				{
					n_forms = bls_sc_batches[schedule].n_terms;
					for ( dim_t form = 0; form < n_forms; ++form )
						term[form] = bls_sc_batches[schedule].term[form];
				}
				else
				{
					n_forms = 1;
					term[0] = term_order[schedule];
				}

				bls_sc_term_exec_t term_exec[BLS_SC_BATCH_CAPACITY];
				const int8_t* a_coeff[BLS_SC_BATCH_CAPACITY] =
				    { NULL, NULL };
				const int8_t* b_coeff[BLS_SC_BATCH_CAPACITY] =
				    { NULL, NULL };

				for ( dim_t form = 0; form < n_forms; ++form )
				{
					term_exec[form] = bls_sc_prepare_term
					(
					  term[form], pp, init_term, direct_c, beta_zero
					);
					a_coeff[form] = bls_sc_terms[term[form]].a;
					b_coeff[form] = bls_sc_terms[term[form]].b;
				}

				const void* b_src[BLS_SC_NQUADS];
				for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
					b_src[q] = bls_sc_const_ptr_offset
					(
					  b_quads[q], pp, jj,
					  rs_b, cs_b, dt_size
					);

				void* b_pack[BLS_SC_BATCH_CAPACITY] = { NULL, NULL };
				inc_t rs_bp, cs_bp, ps_bp;
				if ( pair_batch )
					bls_sc_packm_b_batch
					(
					  dt, kc_alloc, nc_alloc,
					  kc_cur, nc_cur, NR, KR, b_pack_generation,
					  n_forms, b_src, rs_b, cs_b, b_coeff,
					  b_pack, &rs_bp, &cs_bp, &ps_bp,
					  cntx, thread_pb
					);
				else
					bls_sc_packm_b
					(
					  dt, kc_alloc, nc_alloc,
					  kc_cur, nc_cur, NR, KR, b_pack_generation,
					  b_src, rs_b, cs_b, b_coeff[0],
					  &b_pack[0], &rs_bp, &cs_bp, &ps_bp,
					  cntx, thread_pb
					);
				b_pack_generation += 1;
				( void )rs_bp;
				( void )cs_bp;
				ps_bp *= dt_size;

				dim_t ic_start, ic_end;
				const dim_t ic_tid = bli_thrinfo_work_id( thread_ic );
				const dim_t ic_nt  = bli_thrinfo_n_way( thread_ic );
				bli_thread_range_sub
				(
				  ic_tid, ic_nt, mh, MR, FALSE,
				  &ic_start, &ic_end
				);
				const dim_t ic_left = ( ic_end - ic_start ) % MC;
				const dim_t mc_alloc = bli_min( MC, ic_end - ic_start );

				for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
				{
					const dim_t mc_cur =
					    ( MC <= ic_end - ii ? MC : ic_left );
					const void* a_src[BLS_SC_NQUADS];

					for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
						a_src[q] = bls_sc_const_ptr_offset
						(
						  a_quads[q], ii, pp,
						  rs_a, cs_a, dt_size
						);

					void* a_pack[BLS_SC_BATCH_CAPACITY] = { NULL, NULL };
					inc_t rs_ap, cs_ap, ps_ap;
					if ( pair_batch )
						bls_sc_packm_a_batch
						(
						  dt, mc_alloc, kc_alloc,
						  mc_cur, kc_cur, MR, KR, a_pack_generation,
						  n_forms, a_src, rs_a, cs_a, a_coeff,
						  a_pack, &rs_ap, &cs_ap, &ps_ap,
						  cntx, thread_pa
						);
					else
						bls_sc_packm_a
						(
						  dt, mc_alloc, kc_alloc,
						  mc_cur, kc_cur, MR, KR, a_pack_generation,
						  a_src, rs_a, cs_a, a_coeff[0],
						  &a_pack[0], &rs_ap, &cs_ap, &ps_ap,
						  cntx, thread_pa
						);
					a_pack_generation += 1;
					( void )rs_ap;
					( void )cs_ap;
					ps_ap *= dt_size;

					const dim_t jr_nt = bli_thrinfo_n_way( thread_jr );
					const dim_t jr_tid = bli_thrinfo_work_id( thread_jr );
					const dim_t jr_iter = ( nc_cur + NR - 1 ) / NR;
					dim_t jr_start, jr_end;
					bli_thread_range_sub
					(
					  jr_tid, jr_nt, jr_iter, 1, FALSE,
					  &jr_start, &jr_end
					);

					for ( dim_t j = jr_start; j < jr_end; ++j )
					{
						const dim_t nr_cur =
						    bli_min( NR, nc_cur - j * NR );
						const dim_t ir_nt =
						    bli_thrinfo_n_way( thread_ir );
						const dim_t ir_tid =
						    bli_thrinfo_work_id( thread_ir );
						const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR;
						dim_t ir_start, ir_end;
						bli_thread_range_sub
						(
						  ir_tid, ir_nt, ir_iter, 1, FALSE,
						  &ir_start, &ir_end
						);

						for ( dim_t i = ir_start; i < ir_end; ++i )
						{
							const dim_t mr_cur =
							    bli_min( MR, mc_cur - i * MR );
							void* c_tile[BLS_SC_NQUADS];
							const dim_t row = ii + i * MR;
							const dim_t col = jj + j * NR;

							for ( dim_t q = 0; q < BLS_SC_NQUADS; ++q )
								c_tile[q] = bls_sc_ptr_offset
								(
								  c_quads[q], row, col,
								  rs_c, cs_c, dt_size
								);

							for ( dim_t form = 0; form < n_forms; ++form )
							{
								const char* a_ir =
								    ( const char* )a_pack[form] + i * ps_ap;
								const char* b_jr =
								    ( const char* )b_pack[form] + j * ps_bp;
								const char* a2;
								const char* b2;

								if ( form + 1 < n_forms )
								{
									a2 = ( const char* )a_pack[form + 1] +
									     i * ps_ap;
									b2 = ( const char* )b_pack[form + 1] +
									     j * ps_bp;
								}
								else
								{
									const char* a_first =
									    ( const char* )a_pack[0] + i * ps_ap;
									const char* b_first =
									    ( const char* )b_pack[0] + j * ps_bp;
									a2 = bli_gemm_get_next_a_upanel
									     ( a_first, ps_ap, 1 );
									b2 = b_first;

									if ( bli_is_last_iter_slrr
									     ( i, ir_end, ir_tid, ir_nt ) )
									{
										a2 = ( const char* )a_pack[0];
										b2 = bli_gemm_get_next_b_upanel
										     ( b_first, ps_bp, 1 );
										if ( bli_is_last_iter_slrr
										     ( j, jr_end, jr_tid, jr_nt ) )
											b2 = ( const char* )b_pack[0];
									}
								}

								bls_sc_run_term_tile
								(
								  &ukr_ctx, &term_exec[form],
								  kc_cur, mr_cur, nr_cur,
								  a_ir, b_jr, a2, b2, c_tile
								);
							}
						}
					}
				}
			}
		}
	}
}

static err_t bls_sc_gemm_int
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     )
{
	bls_sc_gemm_bp( alpha, a, b, beta, c, cntx, rntm, thread );

	return BLIS_SUCCESS;
}

static void bls_sc_classical_update
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	if ( bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) ||
	     bli_obj_has_zero_dim( c ) ) return;

	bls_sc_default_gemm( alpha, a, b, beta, c, cntx, rntm );
}

void bls_subcuber_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	if ( !bls_sc_env_enabled() ||
	     !bls_sc_operands_supported( a, b, c ) )
	{
		bls_sc_default_gemm( alpha, a, b, beta, c, cntx, rntm );
		return;
	}

	/*
	 * The SubCuber path immediately queries kernel metadata, so materialize
	 * the default native context here. Preserve a caller's null context on
	 * the fallback path, where conventional BLIS may select an induced one.
	 */
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	if ( bli_l3_return_early_if_trivial
	     ( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	rntm_t rntm_base;
	if ( rntm == NULL ) bli_rntm_init_from_global( &rntm_base );
	else                rntm_base = *rntm;

	/*
	 * Explicit IR parallelism currently corrupts nontrivial DGEMM updates in
	 * the fused schedule. Automatic factorization never selects it on this
	 * target. Honor a caller's custom IR split safely through conventional
	 * BLIS until that schedule is made IR-parallel.
	 */
	if ( bli_obj_dt( c ) == BLIS_DOUBLE &&
	     bli_rntm_ir_ways( &rntm_base ) > 1 )
	{
		bls_sc_default_gemm
		( alpha, a, b, beta, c, cntx, &rntm_base );
		return;
	}

	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	const num_t dt = bli_obj_dt( &c_local );
	obj_t alpha_cast;
	obj_t beta_cast;
	obj_t one_cast;
	bli_obj_scalar_init_detached_copy_of
	( dt, BLIS_NO_CONJUGATE, alpha, &alpha_cast );
	bli_obj_scalar_init_detached_copy_of
	( dt, BLIS_NO_CONJUGATE, beta, &beta_cast );
	bli_obj_scalar_init_detached_copy_of
	( dt, BLIS_NO_CONJUGATE, &BLIS_ONE, &one_cast );

	if ( bli_cntx_dislikes_storage_of
	     ( &c_local, BLIS_GEMM_UKR, cntx ) )
	{
		bli_obj_swap( &a_local, &b_local );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

	const dim_t m = bli_obj_length( &c_local );
	const dim_t n = bli_obj_width( &c_local );
	const dim_t k = bli_obj_width( &a_local );
	const gemm_ukr_ft gemm_ukr =
	    bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx );
	const bool armv8a_2c =
	    bls_sc_2c_armv8a_is_compatible( dt, gemm_ukr, cntx );
	const inc_t rs_c = bli_obj_row_stride( &c_local );
	const inc_t cs_c = bli_obj_col_stride( &c_local );
	const bool can_use_2c =
	    armv8a_2c &&
	    bli_cntx_prefers_storage_of( &c_local, BLIS_GEMM_UKR, cntx ) &&
	    rs_c > 0 && cs_c == 1;
	dim_t nt = bli_rntm_num_threads( &rntm_base );
	if ( nt < 1 ) nt = bli_rntm_calc_num_threads( &rntm_base );
	if ( nt < 1 ) nt = 1;

	const dim_t crossover = bls_sc_env_blksz
	(
	  "BLIS_SUBCUBER_MIN_DIM",
	  bls_sc_default_crossover( dt, nt, can_use_2c ),
	  1
	);
	const dim_t problem_min = bli_min( m, bli_min( n, k ) );

	if ( problem_min < crossover )
	{
		bls_sc_default_gemm
		(
		  &alpha_cast, &a_local, &b_local, &beta_cast, &c_local,
		  cntx, &rntm_base
		);
		return;
	}

	const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	dim_t       KR = bli_cntx_get_blksz_def_dt( dt, BLIS_KR, cntx );
	const dim_t PACKMR = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const dim_t PACKNR = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );
	const dim_t BBM = bli_cntx_get_blksz_max_dt( dt, BLIS_BBM, cntx );
	const dim_t BBN = bli_cntx_get_blksz_max_dt( dt, BLIS_BBN, cntx );
	if ( KR == 0 ) KR = 1;

	const dim_t mh = ( ( m / 2 ) / MR ) * MR;
	const dim_t nh = ( ( n / 2 ) / NR ) * NR;
	const dim_t kh = ( ( k / 2 ) / KR ) * KR;

	const dim_t mc = 2 * mh;
	const dim_t nc = 2 * nh;
	const dim_t kc = 2 * kh;

	if ( mc == 0 || nc == 0 || kc == 0 ||
	     PACKMR != MR || PACKNR != NR || BBM != 1 || BBN != 1 ||
	     gemm_ukr == NULL ||
	     MR * NR * bli_dt_size( dt ) > BLIS_STACK_BUF_MAX_SIZE )
	{
		bls_sc_default_gemm
		(
		  &alpha_cast, &a_local, &b_local, &beta_cast, &c_local,
		  cntx, &rntm_base
		);
		return;
	}

	const char* trace = getenv( "BLIS_SUBCUBER_TRACE" );
	if ( trace != NULL && trace[0] != '\0' && trace[0] != '0' )
		fprintf
		(
		  stderr,
		  "blis-subcuber: %s %lldx%lldx%lld core=%lldx%lldx%lld\n",
		  dt == BLIS_FLOAT ? "sgemm" : "dgemm",
		  ( long long )m, ( long long )n, ( long long )k,
		  ( long long )mc, ( long long )nc, ( long long )kc
		);

	obj_t a_core;
	obj_t b_core;
	obj_t c_core;
	bli_acquire_mpart( 0, 0, mc, kc, &a_local, &a_core );
	bli_acquire_mpart( 0, 0, kc, nc, &b_local, &b_core );
	bli_acquire_mpart( 0, 0, mc, nc, &c_local, &c_core );

	rntm_t rntm_fmm = rntm_base;
	bli_rntm_set_pack_a( TRUE, &rntm_fmm );
	bli_rntm_set_pack_b( TRUE, &rntm_fmm );
	/*
	 * All seven products run sequentially on the same team with identical
	 * mh-by-nh output geometry. Their multiplicity scales total work
	 * uniformly and therefore does not change the m:n factorization ratio.
	 * Term-level parallelism would need a separate conflict-aware scheduler,
	 * not a sevenfold dimension passed to the standard factorizer.
	 */
	bli_rntm_factorize( mh, nh, kh, &rntm_fmm );

	/*
	 * SubCuber consumes the standard full-depth SUP tree and validates its
	 * jc->pc->pb->ic->pa->jr->ir shape inside the worker before touching C.
	 * Keep this decorator paired with that contract if BLIS's internal
	 * thread-tree construction changes.
	 */
	bli_l3_sup_thread_decorator
	(
	  bls_sc_gemm_int,
	  BLIS_GEMM,
	  &alpha_cast,
	  &a_core,
	  &b_core,
	  &beta_cast,
	  &c_core,
	  cntx,
	  &rntm_fmm
	);

	if ( kc < k )
	{
		obj_t a_kt;
		obj_t b_kt;
		bli_acquire_mpart( 0,  kc, mc, k - kc, &a_local, &a_kt );
		bli_acquire_mpart( kc, 0,  k - kc, nc, &b_local, &b_kt );
		bls_sc_classical_update
		(
		  &alpha_cast, &a_kt, &b_kt, &one_cast, &c_core,
		  cntx, &rntm_base
		);
	}

	if ( nc < n )
	{
		obj_t a_top;
		obj_t b_right;
		obj_t c_right;
		bli_acquire_mpart( 0, 0,  mc, k,      &a_local, &a_top );
		bli_acquire_mpart( 0, nc, k,  n - nc, &b_local, &b_right );
		bli_acquire_mpart( 0, nc, mc, n - nc, &c_local, &c_right );
		bls_sc_classical_update
		(
		  &alpha_cast, &a_top, &b_right, &beta_cast, &c_right,
		  cntx, &rntm_base
		);
	}

	if ( mc < m )
	{
		obj_t a_bottom;
		obj_t b_all;
		obj_t c_bottom;
		bli_acquire_mpart( mc, 0, m - mc, k, &a_local, &a_bottom );
		bli_acquire_mpart( 0,  0, k,      n, &b_local, &b_all );
		bli_acquire_mpart( mc, 0, m - mc, n, &c_local, &c_bottom );
		bls_sc_classical_update
		(
		  &alpha_cast, &a_bottom, &b_all, &beta_cast, &c_bottom,
		  cntx, &rntm_base
		);
	}
}
