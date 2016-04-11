/*
 * Copyright (c) 2014-2016 Ilya Kaliman <ilya.kaliman@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "aux.h"

#if defined(XM_SCALAR_FLOAT)
#define EPSILON 1.0e-4
#define xm_abs fabsf
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#define EPSILON 1.0e-8
#define xm_abs cabs
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#define EPSILON 1.0e-4
#define xm_abs cabsf
#else /* assume double */
#define EPSILON 1.0e-8
#define xm_abs fabs
#endif

static void
fatal(const char *msg)
{
	fprintf(stdout, "fatal error: %s\n", msg);
	abort();
}

int
main(void)
{
	struct xm_allocator *allocator;
	struct xm_tensor *vvx, *vvy, *oovv, *vvvv, *ref, *res;
	xm_dim_t blk_dim, dim, idx;
	const double alpha = 0.4;
	const size_t block_size = 3, o = 5, v = 9, x = 4;
	size_t i, j, k, l, size;

	printf("test3                                       ");
	fflush(stdout);

	xm_set_log_stream(stderr);
	xm_set_memory_limit(50000);

	if ((allocator = xm_allocator_create(NULL)) == NULL)
		fatal("xm_allocator_create");

	dim = xm_dim_3(v, v, x);
	vvx = xm_tensor_create(allocator, &dim, "vvx");
	dim = xm_dim_3(v, v, x);
	vvy = xm_tensor_create(allocator, &dim, "vvy");
	dim = xm_dim_4(o, o, v, v);
	oovv = xm_tensor_create(allocator, &dim, "oovv");
	dim = xm_dim_4(v, v, v, v);
	vvvv = xm_tensor_create(allocator, &dim, "vvvv");
	dim = xm_dim_4(o, o, v, v);
	ref = xm_tensor_create(allocator, &dim, "ref");
	dim = xm_dim_4(o, o, v, v);
	res = xm_tensor_create(allocator, &dim, "res");

	xm_tensor_init_vvx(vvx, allocator, block_size, XM_INIT_RAND);
	xm_tensor_init_vvx(vvy, allocator, block_size, XM_INIT_RAND);
	xm_tensor_init_oovv(oovv, allocator, block_size, XM_INIT_RAND);
	xm_tensor_init(vvvv, allocator, block_size, XM_INIT_RAND);

	/* compute reference */
	xm_tensor_init_oovv(ref, allocator, block_size, XM_INIT_RAND);
	if (xm_contract(alpha, vvx, vvy, 0.0, vvvv, "daP", "cbP", "abcd"))
		fatal("xm_contract");
	if (xm_contract(1.0, vvvv, oovv, 0.0, ref, "abcd", "ijcd", "ijab"))
		fatal("xm_contract");
	xm_tensor_free(vvvv);

	/* compute result in batches */
	xm_tensor_init_oovv(res, allocator, block_size, XM_INIT_ZERO);
	dim = xm_dim_4(v, v, v, v);
	vvvv = xm_tensor_create(allocator, &dim, "vvvv");
	idx = xm_dim_zero(dim.n);
	blk_dim = xm_dim_same(4, block_size);
	do {
		xm_tensor_set_zero_block(vvvv, &idx, &blk_dim);
	} while (xm_dim_inc(&idx, &dim) == 0);
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++)
		for (k = 0; k < v; k++)
		for (l = 0; l < v; l++) {
			uintptr_t ptr;
			idx = xm_dim_4(j, i, k, l);
			size = xm_dim_dot(&blk_dim) * sizeof(xm_scalar_t);
			ptr = xm_allocator_allocate(allocator, size);
			xm_tensor_reset_block(vvvv, &idx);
			xm_tensor_set_source_block(vvvv, &idx, &blk_dim, ptr);
		}
		if (xm_contract(alpha, vvx, vvy, 0.0, vvvv,
		    "daP", "cbP", "abcd"))
			fatal("xm_contract");
		if (xm_contract(1.0, vvvv, oovv, 1.0, res,
		    "abcd", "ijcd", "ijab"))
			fatal("xm_contract");
		for (j = 0; j < v; j++)
		for (k = 0; k < v; k++)
		for (l = 0; l < v; l++) {
			uintptr_t ptr;
			idx = xm_dim_4(j, i, k, l);
			ptr = xm_tensor_get_block_data_ptr(vvvv, &idx);
			xm_allocator_deallocate(allocator, ptr);
			xm_tensor_reset_block(vvvv, &idx);
			xm_tensor_set_zero_block(vvvv, &idx, &blk_dim);
		}
	}

	/* verify result */
	dim = xm_tensor_get_abs_dim(ref);
	idx = xm_dim_zero(dim.n);
	do {
		xm_scalar_t ref_el, res_el;
		ref_el = xm_tensor_get_abs_element(ref, &idx);
		res_el = xm_tensor_get_abs_element(res, &idx);
		if (xm_abs(ref_el - res_el) > EPSILON)
			fatal("result != reference");
	} while (xm_dim_inc(&idx, &dim) == 0);

	/* free all */
	xm_tensor_free(vvx);
	xm_tensor_free(vvy);
	xm_tensor_free(vvvv);
	xm_tensor_free(oovv);
	xm_tensor_free(ref);
	xm_tensor_free(res);
	xm_allocator_destroy(allocator);

	printf("success\n");
	return (0);
}
