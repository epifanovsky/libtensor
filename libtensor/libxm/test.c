/*
 * Copyright (c) 2017 Ilya Kaliman
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

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef XM_USE_MPI
#include <mpi.h>
#endif

#include "xm.h"
#include "util.h"

typedef void (*test_fn)(const char *, xm_scalar_type_t);
typedef void (*make_ab_fn)(xm_allocator_t *, xm_tensor_t **,
    xm_tensor_t **, xm_scalar_type_t);
typedef void (*make_abc_fn)(xm_allocator_t *, xm_tensor_t **, xm_tensor_t **,
    xm_tensor_t **, xm_scalar_type_t);

struct two_tensor_test {
	make_ab_fn make_ab;
	const char *idxa, *idxb;
};

struct contract_test {
	make_abc_fn make_abc;
	const char *idxa, *idxb, *idxc;
};

static int
scalar_eq(xm_scalar_t a, xm_scalar_t b, xm_scalar_type_t type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float aa = a, bb = b;
		return fabsf(aa - bb) < 1.0e-4;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex aa = a, bb = b;
		return cabsf(aa - bb) < 1.0e-4;
	}
	case XM_SCALAR_DOUBLE: {
		double aa = a, bb = b;
		return fabs(aa - bb) < 1.0e-8;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex aa = a, bb = b;
		return cabs(aa - bb) < 1.0e-8;
	}
	}
	return 0;
}

static xm_scalar_t
random_scalar(xm_scalar_type_t type)
{
	if (type == XM_SCALAR_FLOAT || type == XM_SCALAR_DOUBLE)
		return (drand48() - 0.5);
	return (drand48() - 0.5) + (drand48() - 0.5) * I;
}

static void
fill_random(xm_tensor_t *t)
{
	xm_dim_t idx, nblocks;
	size_t i, blksize;
	void *buf;
	xm_block_type_t blocktype;
	xm_scalar_type_t type;

	type = xm_tensor_get_scalar_type(t);
	buf = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf);
	nblocks = xm_tensor_get_nblocks(t);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		blocktype = xm_tensor_get_block_type(t, idx);
		if (blocktype == XM_BLOCK_TYPE_CANONICAL) {
			blksize = xm_tensor_get_block_size(t, idx);
			switch (type) {
			case XM_SCALAR_FLOAT: {
				float *xbuf = buf;
				for (i = 0; i < blksize; i++)
					xbuf[i] = random_scalar(type);
				break;
			}
			case XM_SCALAR_FLOAT_COMPLEX: {
				float complex *xbuf = buf;
				for (i = 0; i < blksize; i++)
					xbuf[i] = random_scalar(type);
				break;
			}
			case XM_SCALAR_DOUBLE: {
				double *xbuf = buf;
				for (i = 0; i < blksize; i++)
					xbuf[i] = random_scalar(type);
				break;
			}
			case XM_SCALAR_DOUBLE_COMPLEX: {
				double complex *xbuf = buf;
				for (i = 0; i < blksize; i++)
					xbuf[i] = random_scalar(type);
				break;
			}
			}
			xm_tensor_write_block(t, idx, buf);
		}
		xm_dim_inc(&idx, &nblocks);
	}
	free(buf);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

static void
compare_tensors(xm_tensor_t *t, xm_tensor_t *u)
{
	xm_dim_t idx, dimst, dimsu;

	dimst = xm_tensor_get_abs_dims(t);
	dimsu = xm_tensor_get_abs_dims(u);
	assert(xm_dim_eq(&dimst, &dimsu));
	idx = xm_dim_zero(dimst.n);
	while (xm_dim_ne(&idx, &dimst)) {
		xm_scalar_t et = xm_tensor_get_element(t, idx);
		xm_scalar_t eu = xm_tensor_get_element(u, idx);
		if (!scalar_eq(et, eu, xm_tensor_get_scalar_type(t)))
			fatal("tensors do not match");
		xm_dim_inc(&idx, &dimst);
	}
}

static void
check_add(xm_tensor_t *aa, xm_scalar_t alpha, xm_tensor_t *a, xm_scalar_t beta,
    xm_tensor_t *b, const char *idxa, const char *idxb)
{
	xm_dim_t absdimsa, absdimsb, cidxa, cidxb, ia, ib;
	xm_scalar_t eaa, ref;

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	ia = xm_dim_zero(absdimsa.n);
	ib = xm_dim_zero(absdimsb.n);
	while (xm_dim_ne(&ia, &absdimsa)) {
		xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
		ref = alpha * xm_tensor_get_element(a, ia) +
		    beta * xm_tensor_get_element(b, ib);
		eaa = xm_tensor_get_element(aa, ia);
		if (!scalar_eq(eaa, ref, xm_tensor_get_scalar_type(aa)))
			fatal("result != reference");
		xm_dim_inc(&ia, &absdimsa);
	}
}

static void
check_mul(xm_tensor_t *aa, xm_tensor_t *a, xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	xm_dim_t absdimsa, absdimsb, cidxa, cidxb, ia, ib;
	xm_scalar_t eaa, ref;

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	ia = xm_dim_zero(absdimsa.n);
	ib = xm_dim_zero(absdimsb.n);
	while (xm_dim_ne(&ia, &absdimsa)) {
		xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
		ref = xm_tensor_get_element(a, ia) *
		      xm_tensor_get_element(b, ib);
		eaa = xm_tensor_get_element(aa, ia);
		if (!scalar_eq(eaa, ref, xm_tensor_get_scalar_type(aa)))
			fatal("result != reference");
		xm_dim_inc(&ia, &absdimsa);
	}
}

static void
check_div(xm_tensor_t *aa, xm_tensor_t *a, xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	xm_dim_t absdimsa, absdimsb, cidxa, cidxb, ia, ib;
	xm_scalar_t eaa, ref;

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	ia = xm_dim_zero(absdimsa.n);
	ib = xm_dim_zero(absdimsb.n);
	while (xm_dim_ne(&ia, &absdimsa)) {
		xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
		ref = xm_tensor_get_element(a, ia) /
		      xm_tensor_get_element(b, ib);
		eaa = xm_tensor_get_element(aa, ia);
		if (!scalar_eq(eaa, ref, xm_tensor_get_scalar_type(aa)))
			fatal("result != reference");
		xm_dim_inc(&ia, &absdimsa);
	}
}

static void
check_dot(xm_scalar_t res, xm_tensor_t *a, xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	xm_dim_t absdimsa, absdimsb, cidxa, cidxb, ia, ib;
	xm_scalar_t ref = 0;

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	ia = xm_dim_zero(absdimsa.n);
	ib = xm_dim_zero(absdimsb.n);
	while (xm_dim_ne(&ia, &absdimsa)) {
		xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
		ref += xm_tensor_get_element(a, ia) *
		       xm_tensor_get_element(b, ib);
		xm_dim_inc(&ia, &absdimsa);
	}
	if (!scalar_eq(res, ref, xm_tensor_get_scalar_type(a)))
		fatal("result != reference");
}

static void
check_contract(xm_tensor_t *cc, xm_scalar_t alpha, xm_tensor_t *a,
    xm_tensor_t *b, xm_scalar_t beta, xm_tensor_t *c, const char *idxa,
    const char *idxb, const char *idxc)
{
	xm_dim_t absdimsa, absdimsb, absdimsc, ia, ib, ic;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc;
	xm_scalar_t ref, ecc;
	size_t k, nk;

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	xm_make_masks(idxc, idxa, &cidxc, &aidxa);
	xm_make_masks(idxc, idxb, &aidxc, &aidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	absdimsc = xm_tensor_get_abs_dims(c);
	nk = xm_dim_dot_mask(&absdimsa, &cidxa);
	ic = xm_dim_zero(absdimsc.n);
	while (xm_dim_ne(&ic, &absdimsc)) {
		ref = beta * xm_tensor_get_element(c, ic);
		ia = xm_dim_zero(absdimsa.n);
		ib = xm_dim_zero(absdimsb.n);
		xm_dim_set_mask(&ia, &aidxa, &ic, &cidxc);
		xm_dim_set_mask(&ib, &aidxb, &ic, &aidxc);
		for (k = 0; k < nk; k++) {
			xm_scalar_t ea = xm_tensor_get_element(a, ia);
			xm_scalar_t eb = xm_tensor_get_element(b, ib);
			ref += alpha * ea * eb;
			xm_dim_inc_mask(&ia, &absdimsa, &cidxa);
			xm_dim_inc_mask(&ib, &absdimsb, &cidxb);
		}
		ecc = xm_tensor_get_element(cc, ic);
		if (!scalar_eq(ecc, ref, xm_tensor_get_scalar_type(cc)))
			fatal("result != reference");
		xm_dim_inc(&ic, &absdimsc);
	}
}

static void
test_add(const struct two_tensor_test *test, const char *path,
    xm_scalar_type_t type, xm_scalar_t alpha, xm_scalar_t beta)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *aa;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_ab(allocator, &a, &b, type);
	assert(a);
	assert(b);
	fill_random(a);
	fill_random(b);
	aa = xm_tensor_create_structure(a, type, allocator);
	xm_copy(aa, 1, a, test->idxa, test->idxa);
	xm_add(alpha, aa, beta, b, test->idxa, test->idxb);
	check_add(aa, alpha, a, beta, b, test->idxa, test->idxb);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(aa);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(aa);
	xm_allocator_destroy(allocator);
}

static void
test_mul(const struct two_tensor_test *test, const char *path,
    xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *aa;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_ab(allocator, &a, &b, type);
	assert(a);
	assert(b);
	fill_random(a);
	fill_random(b);
	aa = xm_tensor_create_structure(a, type, allocator);
	xm_copy(aa, 1, a, test->idxa, test->idxa);
	xm_mul(aa, b, test->idxa, test->idxb);
	check_mul(aa, a, b, test->idxa, test->idxb);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(aa);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(aa);
	xm_allocator_destroy(allocator);
}

static void
test_div(const struct two_tensor_test *test, const char *path,
    xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *aa;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_ab(allocator, &a, &b, type);
	assert(a);
	assert(b);
	fill_random(a);
	fill_random(b);
	aa = xm_tensor_create_structure(a, type, allocator);
	xm_copy(aa, 1, a, test->idxa, test->idxa);
	xm_div(aa, b, test->idxa, test->idxb);
	check_div(aa, a, b, test->idxa, test->idxb);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(aa);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(aa);
	xm_allocator_destroy(allocator);
}

static void
test_dot(const struct two_tensor_test *test, const char *path,
    xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b;
	xm_scalar_t res;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_ab(allocator, &a, &b, type);
	assert(a);
	assert(b);
	fill_random(a);
	fill_random(b);
	res = xm_dot(a, b, test->idxa, test->idxb);
	check_dot(res, a, b, test->idxa, test->idxb);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_contract(const struct contract_test *test, const char *path,
    xm_scalar_type_t type, xm_scalar_t alpha, xm_scalar_t beta)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *c, *cc;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_abc(allocator, &a, &b, &c, type);
	assert(a);
	assert(b);
	assert(c);
	fill_random(a);
	fill_random(b);
	fill_random(c);
	cc = xm_tensor_create_structure(c, type, allocator);
	xm_copy(cc, 1, c, test->idxc, test->idxc);
	xm_contract(alpha, a, b, beta, cc, test->idxa, test->idxb, test->idxc);
	check_contract(cc, alpha, a, b, beta, c, test->idxa, test->idxb,
	    test->idxc);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(c);
	xm_tensor_free_block_data(cc);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_tensor_free(cc);
	xm_allocator_destroy(allocator);
}

static void
make_ab_1(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_scalar_type_t type)
{
	xm_block_space_t *bs;
	xm_tensor_t *a, *b;

	bs = xm_block_space_create(xm_dim_8(1,2,3,4,5,6,7,8));
	xm_block_space_split(bs, 1, 1);
	xm_block_space_split(bs, 3, 2);
	xm_block_space_split(bs, 5, 3);
	xm_block_space_split(bs, 7, 2);
	xm_block_space_split(bs, 7, 5);
	a = xm_tensor_create_canonical(bs, type, allocator);
	b = xm_tensor_create_canonical(bs, type, allocator);
	xm_block_space_free(bs);
	*aa = a;
	*bb = b;
}

static void
make_ab_2(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_scalar_type_t type)
{
	xm_block_space_t *bs;
	xm_tensor_t *a, *b;

	bs = xm_block_space_create(xm_dim_4(4,4,4,4));
	xm_block_space_split(bs, 0, 2);
	xm_block_space_split(bs, 1, 2);
	xm_block_space_split(bs, 2, 2);
	xm_block_space_split(bs, 3, 2);
	a = xm_tensor_create_canonical(bs, type, allocator);
	b = xm_tensor_create_canonical(bs, type, allocator);
	xm_block_space_free(bs);
	*aa = a;
	*bb = b;
}

static void
make_ab_3(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_scalar_type_t type)
{
	xm_block_space_t *bs;
	xm_tensor_t *a, *b;

	bs = xm_block_space_create(xm_dim_2(8, 8));
	xm_block_space_split(bs, 0, 4);
	xm_block_space_split(bs, 1, 4);
	a = xm_tensor_create_canonical(bs, type, allocator);
	b = xm_tensor_create(bs, type, allocator);
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(b, xm_dim_2(1, 1));
	xm_tensor_set_derivative_block(b, xm_dim_2(0, 1), xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -4);
	xm_block_space_free(bs);
	*aa = a;
	*bb = b;
}

static void
make_ab_4(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_scalar_type_t type)
{
	xm_block_space_t *bs;
	xm_tensor_t *a, *b;

	bs = xm_block_space_create(xm_dim_2(20, 20));
	xm_block_space_split(bs, 0, 10);
	xm_block_space_split(bs, 1, 10);
	a = xm_tensor_create(bs, type, allocator);
	xm_tensor_set_canonical_block(a, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(a, xm_dim_2(1, 1));
	b = xm_tensor_create(bs, type, allocator);
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 0));
	xm_tensor_set_derivative_block(b, xm_dim_2(1, 1), xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -10);
	xm_block_space_free(bs);
	*aa = a;
	*bb = b;
}

static void
make_ab_5(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_scalar_type_t type)
{
	xm_block_space_t *bs;
	xm_tensor_t *a, *b;

	bs = xm_block_space_create(xm_dim_2(17, 31));
	xm_block_space_split(bs, 0, 11);
	xm_block_space_split(bs, 1, 15);
	a = xm_tensor_create(bs, type, allocator);
	xm_tensor_set_canonical_block(a, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(a, xm_dim_2(1, 1));
	b = xm_tensor_create(bs, type, allocator);
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(b, xm_dim_2(1, 1));
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 1));
	xm_tensor_set_canonical_block(b, xm_dim_2(1, 0));
	xm_block_space_free(bs);
	*aa = a;
	*bb = b;
}

static void
make_abc_1(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create(bsa, type, allocator);
	c = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_2(0, 0);
	xm_tensor_set_canonical_block(a, idx);
	xm_tensor_set_canonical_block(b, idx);
	xm_tensor_set_canonical_block(c, idx);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_2(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, nblocks;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create(bsa, type, allocator);
	c = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	nblocks = xm_tensor_get_nblocks(a);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			fatal("unexpected block type");
		xm_tensor_set_canonical_block(a, idx);
		xm_tensor_set_canonical_block(b, idx);
		xm_tensor_set_canonical_block(c, idx);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_3(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_block_space_t *bsa, *bsc;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_same(8, 1));
	a = xm_tensor_create_canonical(bsa, type, allocator);
	b = xm_tensor_create_canonical(bsa, type, allocator);
	xm_block_space_free(bsa);
	bsc = xm_block_space_create(xm_dim_same(2, 1));
	c = xm_tensor_create_canonical(bsc, type, allocator);
	xm_block_space_free(bsc);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_4(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_same(6, 3));
	a = xm_tensor_create_canonical(bsa, type, allocator);
	b = xm_tensor_create_canonical(bsa, type, allocator);
	c = xm_tensor_create_canonical(bsa, type, allocator);
	xm_block_space_free(bsa);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_5(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create(bsa, type, allocator);
	c = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_2(0, 0);
	xm_tensor_set_canonical_block(a, idx);
	xm_tensor_set_canonical_block(b, idx);
	xm_tensor_set_canonical_block(c, idx);
	idx = xm_dim_2(0, 1);
	xm_tensor_set_canonical_block(c, idx);
	idx = xm_dim_2(1, 0);
	xm_tensor_set_canonical_block(c, idx);
	idx = xm_dim_2(1, 1);
	xm_tensor_set_canonical_block(a, idx);
	xm_tensor_set_canonical_block(b, idx);
	xm_tensor_set_canonical_block(c, idx);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_6(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_1(3));
	xm_block_space_split(bsa, 0, 1);
	a = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	bsb = xm_block_space_create(xm_dim_2(2, 5));
	xm_block_space_split(bsb, 1, 3);
	b = xm_tensor_create(bsb, type, allocator);
	xm_block_space_free(bsb);
	bsc = xm_block_space_create(xm_dim_3(2, 3, 5));
	xm_block_space_split(bsc, 1, 1);
	xm_block_space_split(bsc, 2, 3);
	c = xm_tensor_create(bsc, type, allocator);
	xm_block_space_free(bsc);

	xm_tensor_set_canonical_block(a, xm_dim_1(0));
	xm_tensor_set_canonical_block(a, xm_dim_1(1));
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(b, xm_dim_2(0, 1));
	xm_tensor_set_canonical_block(c, xm_dim_3(0, 0, 0));
	xm_tensor_set_canonical_block(c, xm_dim_3(0, 0, 1));
	xm_tensor_set_canonical_block(c, xm_dim_3(0, 1, 0));
	xm_tensor_set_canonical_block(c, xm_dim_3(0, 1, 1));

	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_7(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, nblocks;
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_3(7, 4, 11));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 0, 5);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 2, 9);
	a = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	xm_tensor_set_canonical_block(a, xm_dim_3(0, 0, 0));
	xm_tensor_set_canonical_block(a, xm_dim_3(1, 0, 0));
	xm_tensor_set_derivative_block(a, xm_dim_3(2, 0, 0), xm_dim_3(0, 0, 0),
	    xm_dim_3(2, 1, 0), -0.5);
	xm_tensor_set_derivative_block(a, xm_dim_3(0, 1, 0), xm_dim_3(0, 0, 0),
	    xm_dim_3(1, 0, 2), 1.5);
	xm_tensor_set_canonical_block(a, xm_dim_3(1, 1, 0));
	xm_tensor_set_derivative_block(a, xm_dim_3(2, 1, 0), xm_dim_3(0, 0, 0),
	    xm_dim_3(2, 0, 1), 0.7);
	nblocks = xm_tensor_get_nblocks(a);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_dim_t tt[] = { xm_dim_3(0, 0, 0),
					  xm_dim_3(2, 1, 0) };
			if (xm_dim_eq(&idx, &tt[0]) ||
			    xm_dim_eq(&idx, &tt[1]))
				fatal("unexpected block type");
			xm_tensor_set_canonical_block(a, idx);
		}
	}

	bsb = xm_block_space_create(xm_dim_3(7, 4, 8));
	xm_block_space_split(bsb, 0, 2);
	xm_block_space_split(bsb, 0, 5);
	xm_block_space_split(bsb, 1, 2);
	xm_block_space_split(bsb, 2, 2);
	xm_block_space_split(bsb, 2, 6);
	b = xm_tensor_create_canonical(bsb, type, allocator);
	xm_block_space_free(bsb);

	bsc = xm_block_space_create(xm_dim_2(8, 11));
	xm_block_space_split(bsc, 0, 2);
	xm_block_space_split(bsc, 0, 6);
	xm_block_space_split(bsc, 1, 2);
	xm_block_space_split(bsc, 1, 9);
	c = xm_tensor_create_canonical(bsc, type, allocator);
	xm_block_space_free(bsc);

	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_8(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;

	bsa = xm_block_space_create(xm_dim_2(15, 10));
	xm_block_space_split(bsa, 0, 5);
	xm_block_space_split(bsa, 0, 10);
	xm_block_space_split(bsa, 1, 5);
	a = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	xm_tensor_set_canonical_block(a, xm_dim_2(0, 0));
	xm_tensor_set_derivative_block(a, xm_dim_2(0, 1), xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -0.3);
	xm_tensor_set_canonical_block(a, xm_dim_2(2, 0));
	xm_tensor_set_canonical_block(a, xm_dim_2(2, 1));

	bsb = xm_block_space_create(xm_dim_1(10));
	xm_block_space_split(bsb, 0, 5);
	b = xm_tensor_create(bsb, type, allocator);
	xm_block_space_free(bsb);
	xm_tensor_set_canonical_block(b, xm_dim_1(0));
	xm_tensor_set_canonical_block(b, xm_dim_1(1));

	bsc = xm_block_space_create(xm_dim_1(15));
	xm_block_space_split(bsc, 0, 5);
	xm_block_space_split(bsc, 0, 10);
	c = xm_tensor_create(bsc, type, allocator);
	xm_block_space_free(bsc);
	xm_tensor_set_canonical_block(c, xm_dim_1(0));
	xm_tensor_set_canonical_block(c, xm_dim_1(2));

	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_9(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b, *c;
	const size_t o = 6, v = 10;
	const size_t nblko = 3, nblkv = 4;

	bsa = xm_block_space_create(xm_dim_4(o, o, v, v));
	bsb = xm_block_space_create(xm_dim_4(v, v, v, v));

	/* o: 2, 3, 1 = 6 */
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 0, 5);
	xm_block_space_split(bsa, 1, 5);

	/* v: 3, 1, 4, 2 = 10 */
	xm_block_space_split(bsa, 2, 3);
	xm_block_space_split(bsa, 3, 3);
	xm_block_space_split(bsb, 0, 3);
	xm_block_space_split(bsb, 1, 3);
	xm_block_space_split(bsb, 2, 3);
	xm_block_space_split(bsb, 3, 3);

	xm_block_space_split(bsa, 2, 4);
	xm_block_space_split(bsa, 3, 4);
	xm_block_space_split(bsb, 0, 4);
	xm_block_space_split(bsb, 1, 4);
	xm_block_space_split(bsb, 2, 4);
	xm_block_space_split(bsb, 3, 4);

	xm_block_space_split(bsa, 2, 8);
	xm_block_space_split(bsa, 3, 8);
	xm_block_space_split(bsb, 0, 8);
	xm_block_space_split(bsb, 1, 8);
	xm_block_space_split(bsb, 2, 8);
	xm_block_space_split(bsb, 3, 8);

	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create(bsb, type, allocator);
	c = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	xm_block_space_free(bsb);

	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblko; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblko; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		xm_tensor_set_canonical_block(a, idx);
		xm_tensor_set_canonical_block(c, idx);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, 1);
		}
	}

	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblkv; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblkv; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(b, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		xm_tensor_set_canonical_block(b, idx);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[0], idx.i[1]);
		perm = xm_dim_4(2, 3, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[0], idx.i[1]);
		perm = xm_dim_4(3, 2, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[1], idx.i[0]);
		perm = xm_dim_4(2, 3, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[1], idx.i[0]);
		perm = xm_dim_4(3, 2, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_10(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b, *c;
	const size_t o = 6, v = 9;
	const size_t nblko = 2, nblkv = 3;

	bsa = xm_block_space_create(xm_dim_4(o, o, v, v));
	xm_block_space_split(bsa, 0, 4);
	xm_block_space_split(bsa, 1, 4);
	xm_block_space_split(bsa, 2, 3);
	xm_block_space_split(bsa, 3, 3);
	xm_block_space_split(bsa, 2, 7);
	xm_block_space_split(bsa, 3, 7);
	a = xm_tensor_create(bsa, type, allocator);
	c = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblko; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblko; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		xm_tensor_set_canonical_block(a, idx);
		xm_tensor_set_canonical_block(c, idx);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(a, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, 1);
		}
	}

	bsb = xm_block_space_create(xm_dim_4(v, v, v, v));
	xm_block_space_split(bsb, 0, 3);
	xm_block_space_split(bsb, 1, 3);
	xm_block_space_split(bsb, 2, 3);
	xm_block_space_split(bsb, 3, 3);
	xm_block_space_split(bsb, 0, 7);
	xm_block_space_split(bsb, 1, 7);
	xm_block_space_split(bsb, 2, 7);
	xm_block_space_split(bsb, 3, 7);
	b = xm_tensor_create(bsb, type, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblkv; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblkv; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(b, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		xm_tensor_set_canonical_block(b, idx);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[0], idx.i[1]);
		perm = xm_dim_4(2, 3, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[0], idx.i[1]);
		perm = xm_dim_4(3, 2, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[1], idx.i[0]);
		perm = xm_dim_4(2, 3, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[1], idx.i[0]);
		perm = xm_dim_4(3, 2, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
	}
	*aa = b; /* swap */
	*bb = a; /* swap */
	*cc = c;
}

static void
make_abc_11(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b;

	bsa = xm_block_space_create(xm_dim_4(6, 6, 6, 6));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 3, 2);
	a = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_4(0, 0, 0, 0);
	xm_tensor_set_canonical_block(a, idx);
	idx = xm_dim_4(1, 0, 0, 0);
	xm_tensor_set_canonical_block(a, idx);
	idx = xm_dim_4(0, 1, 0, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 0, 0);
	xm_tensor_set_canonical_block(a, idx);
	idx = xm_dim_4(0, 0, 1, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 0, 1, 0);
	xm_tensor_set_canonical_block(a, idx);
	idx = xm_dim_4(0, 1, 1, 0);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 1, 0);
	xm_tensor_set_canonical_block(a, idx);
	idx = xm_dim_4(0, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 1, 0);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(0, 1, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 1, 0, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(0, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(0, 1, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(3, 2, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 1, 1);
	xm_tensor_set_canonical_block(a, idx);

	bsb = xm_block_space_create(xm_dim_4(3, 3, 6, 6));
	xm_block_space_split(bsb, 2, 2);
	xm_block_space_split(bsb, 3, 2);
	b = xm_tensor_create(bsb, type, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_4(0, 0, 0, 0);
	xm_tensor_set_canonical_block(b, idx);
	idx = xm_dim_4(0, 0, 0, 1);
	xm_tensor_set_canonical_block(b, idx);
	idx = xm_dim_4(0, 0, 1, 0);
	xm_tensor_set_canonical_block(b, idx);
	idx = xm_dim_4(0, 0, 1, 1);
	xm_tensor_set_canonical_block(b, idx);

	*aa = a;
	*bb = b;
	*cc = xm_tensor_create_structure(b, type, allocator);
}

static void
make_abc_12(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc, xm_scalar_type_t type)
{
	xm_dim_t idx, idx2, perm, nblocks;
	xm_block_space_t *bsa, *bsc;
	xm_tensor_t *a, *b, *c;
	const size_t v = 11;

	bsa = xm_block_space_create(xm_dim_4(v, v, v, v));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 3, 2);
	xm_block_space_split(bsa, 0, 7);
	xm_block_space_split(bsa, 1, 7);
	xm_block_space_split(bsa, 2, 7);
	xm_block_space_split(bsa, 3, 7);
	xm_block_space_split(bsa, 0, 8);
	xm_block_space_split(bsa, 1, 8);
	xm_block_space_split(bsa, 2, 8);
	xm_block_space_split(bsa, 3, 8);
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create(bsa, type, allocator);
	xm_block_space_free(bsa);

	bsc = xm_block_space_create(xm_dim_2(v, v));
	xm_block_space_split(bsc, 0, 2);
	xm_block_space_split(bsc, 1, 2);
	xm_block_space_split(bsc, 0, 7);
	xm_block_space_split(bsc, 1, 7);
	xm_block_space_split(bsc, 0, 8);
	xm_block_space_split(bsc, 1, 8);
	c = xm_tensor_create(bsc, type, allocator);
	xm_block_space_free(bsc);

	idx = xm_dim_zero(4);
	nblocks = xm_tensor_get_nblocks(b);
	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_type(b, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		xm_tensor_set_canonical_block(a, idx);
		xm_tensor_set_canonical_block(b, idx);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		}
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[0], idx.i[1]);
		perm = xm_dim_4(2, 3, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		}
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[0], idx.i[1]);
		perm = xm_dim_4(3, 2, 0, 1);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[1], idx.i[0]);
		perm = xm_dim_4(2, 3, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[1], idx.i[0]);
		perm = xm_dim_4(3, 2, 1, 0);
		if (xm_tensor_get_block_type(b, idx2) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		}
	}

	idx = xm_dim_zero(2);
	nblocks = xm_tensor_get_nblocks(c);
	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
		xm_tensor_set_canonical_block(c, idx);
	}

	*aa = a;
	*bb = b;
	*cc = c;
}

static void
test_unfold_1(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	void *buf1, *buf2;

	allocator = xm_allocator_create(path);
	assert(allocator);
	bs = xm_block_space_create(xm_dim_1(15));
	assert(bs);
	xm_block_space_split(bs, 0, 5);
	xm_block_space_split(bs, 0, 10);
	t = xm_tensor_create(bs, type, allocator);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	xm_tensor_set_canonical_block(t, xm_dim_1(0));
	xm_tensor_set_derivative_block(t, xm_dim_1(1), xm_dim_1(0),
	    xm_dim_identity_permutation(1), 0.5);
	fill_random(t);
	u = xm_tensor_create_structure(t, type, allocator);
	xm_copy(u, 1, t, "i", "i");
	buf1 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf1);
	buf2 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf2);
	xm_tensor_read_block(t, xm_dim_1(1), buf1);
	xm_tensor_unfold_block(t, xm_dim_1(1), xm_dim_1(0), xm_dim_zero(0),
	    buf1, buf2, 5);
	xm_tensor_write_block(t, xm_dim_1(0), buf2);
	compare_tensors(t, u);

	xm_tensor_unfold_block(t, xm_dim_1(0), xm_dim_1(0), xm_dim_zero(0),
	    buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_1(0), xm_dim_1(0), xm_dim_zero(0),
	    buf2, buf1, 5);
	xm_tensor_write_block(t, xm_dim_1(0), buf1);
	compare_tensors(t, u);
	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator);
}

static void
test_unfold_2(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	void *buf1, *buf2;

	allocator = xm_allocator_create(path);
	assert(allocator);
	bs = xm_block_space_create(xm_dim_2(5, 10));
	assert(bs);
	xm_block_space_split(bs, 1, 5);
	t = xm_tensor_create(bs, type, allocator);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	xm_tensor_set_canonical_block(t, xm_dim_2(0, 0));
	xm_tensor_set_derivative_block(t, xm_dim_2(0, 1), xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -0.3);
	fill_random(t);
	u = xm_tensor_create_structure(t, type, allocator);
	xm_copy(u, 1, t, "ij", "ij");
	buf1 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf1);
	buf2 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf2);
	xm_tensor_read_block(t, xm_dim_2(0, 1), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 1), xm_dim_1(1), xm_dim_1(0),
	    buf1, buf2, 5);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf2);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_2(0, 1),
	    xm_dim_zero(0), buf1, buf2, 25);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_2(0, 1),
	    xm_dim_zero(0), buf2, buf1, 25);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_2(1, 0),
	    xm_dim_zero(0), buf1, buf2, 25);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_2(1, 0),
	    xm_dim_zero(0), buf2, buf1, 25);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_1(0),
	    xm_dim_1(1), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_1(0),
	    xm_dim_1(1), buf2, buf1, 5);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_1(1),
	    xm_dim_1(0), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_1(1),
	    xm_dim_1(0), buf2, buf1, 5);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(0, 1), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(0, 1), buf2, buf1, 1);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_2(0, 0), buf1);
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(1, 0), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(1, 0), buf2, buf1, 1);
	xm_tensor_write_block(t, xm_dim_2(0, 0), buf1);
	compare_tensors(t, u);

	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator);
}

static void
test_unfold_3(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	void *buf1, *buf2;

	allocator = xm_allocator_create(path);
	assert(allocator);
	bs = xm_block_space_create(xm_dim_4(3, 4, 5, 6));
	assert(bs);
	t = xm_tensor_create(bs, type, allocator);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	xm_tensor_set_canonical_block(t, xm_dim_zero(4));
	fill_random(t);
	u = xm_tensor_create_structure(t, type, allocator);
	xm_copy(u, 1, t, "ijab", "ijab");
	buf1 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf1);
	buf2 = malloc(xm_tensor_get_largest_block_bytes(t));
	assert(buf2);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_4(0, 1, 2, 3),
	    xm_dim_zero(0), buf1, buf2, 3*4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_4(0, 1, 2, 3),
	    xm_dim_zero(0), buf2, buf1, 3*4*5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_4(2, 0, 1, 3),
	    xm_dim_zero(0), buf1, buf2, 3*4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_4(2, 0, 1, 3),
	    xm_dim_zero(0), buf2, buf1, 3*4*5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_3(1, 3, 2),
	    xm_dim_1(0), buf1, buf2, 4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_3(1, 3, 2),
	    xm_dim_1(0), buf2, buf1, 4*5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_3(3, 2, 1),
	    xm_dim_1(0), buf1, buf2, 4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_3(3, 2, 1),
	    xm_dim_1(0), buf2, buf1, 4*5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(1, 0), buf1, buf2, 5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(1, 0), buf2, buf1, 5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(0, 1), buf1, buf2, 5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(0, 1), buf2, buf1, 5*6);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(3, 0, 1), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(3, 0, 1), buf2, buf1, 5);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(1, 3, 0), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(1, 3, 0), buf2, buf1, 5);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 1, 2, 3), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 1, 2, 3), buf2, buf1, 1);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 3, 2, 1), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 3, 2, 1), buf2, buf1, 1);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	xm_tensor_read_block(t, xm_dim_zero(4), buf1);
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(1, 3, 2, 0), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(1, 3, 2, 0), buf2, buf1, 1);
	xm_tensor_write_block(t, xm_dim_zero(4), buf1);
	compare_tensors(t, u);

	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator);
}

static void
test_copy_1(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *c;
	xm_block_space_t *bs;
	xm_dim_t dims, idx;
	const xm_scalar_t sb = random_scalar(type);
	const xm_scalar_t sc = random_scalar(type);

	allocator = xm_allocator_create(path);
	dims = xm_dim_7(3, 4, 1, 7, 3, 5, 9);
	bs = xm_block_space_create(dims);
	assert(bs);
	xm_block_space_split(bs, 0, 1);
	xm_block_space_split(bs, 3, 2);
	xm_block_space_split(bs, 3, 4);
	xm_block_space_split(bs, 4, 2);
	xm_block_space_split(bs, 5, 3);
	xm_block_space_split(bs, 6, 2);
	xm_block_space_split(bs, 6, 6);
	a = xm_tensor_create_canonical(bs, type, allocator);
	xm_block_space_free(bs);
	fill_random(a);
	b = xm_tensor_create_structure(a, type, allocator);
	xm_copy(b, sb, a, "1234567", "1234567");
	c = xm_tensor_create_structure(a, type, allocator);
	xm_copy(c, sc, a, "1234567", "1234567");
	xm_copy(c, sb, c, "1234567", "1234567");
	idx = xm_dim_zero(dims.n);
	while (xm_dim_ne(&idx, &dims)) {
		xm_scalar_t aa = xm_tensor_get_element(a, idx);
		xm_scalar_t bb = xm_tensor_get_element(b, idx);
		xm_scalar_t cc = xm_tensor_get_element(c, idx);
		if (!scalar_eq(aa*sb, bb, type))
			fatal("tensors do not match");
		if (!scalar_eq(aa*sb*sc, cc, type))
			fatal("tensors do not match");
		xm_dim_inc(&idx, &dims);
	}
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(c);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_allocator_destroy(allocator);
}

static void
test_copy_2(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b;
	xm_block_space_t *bs;
	xm_dim_t dims, ia, ib;
	const xm_scalar_t s = random_scalar(type);

	allocator = xm_allocator_create(path);
	dims = xm_dim_2(43, 43);
	bs = xm_block_space_create(dims);
	assert(bs);
	xm_block_space_split(bs, 0, 4);
	xm_block_space_split(bs, 0, 7);
	xm_block_space_split(bs, 0, 21);
	xm_block_space_split(bs, 0, 27);
	xm_block_space_split(bs, 0, 38);
	xm_block_space_split(bs, 1, 4);
	xm_block_space_split(bs, 1, 7);
	xm_block_space_split(bs, 1, 21);
	xm_block_space_split(bs, 1, 27);
	xm_block_space_split(bs, 1, 38);
	a = xm_tensor_create_canonical(bs, type, allocator);
	xm_block_space_free(bs);
	fill_random(a);
	b = xm_tensor_create_structure(a, type, allocator);
	xm_copy(b, s, a, "ji", "ij");
	ia = xm_dim_zero(dims.n);
	while (xm_dim_ne(&ia, &dims)) {
		xm_scalar_t aa, bb;
		ib = xm_dim_2(ia.i[1], ia.i[0]);
		aa = xm_tensor_get_element(a, ia);
		bb = xm_tensor_get_element(b, ib);
		if (!scalar_eq(aa*s, bb, type))
			fatal("tensors do not match");
		xm_dim_inc(&ia, &dims);
	}
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_copy_3(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b;
	xm_block_space_t *bsa, *bsb;
	xm_dim_t dims, ia, ib;
	const xm_dim_t permutation = xm_dim_3(2, 0, 1);
	const xm_scalar_t s = random_scalar(type);

	allocator = xm_allocator_create(path);
	dims = xm_dim_3(43, 31, 28);
	bsa = xm_block_space_create(dims);
	assert(bsa);
	xm_block_space_split(bsa, 0, 4);
	xm_block_space_split(bsa, 0, 7);
	xm_block_space_split(bsa, 0, 21);
	xm_block_space_split(bsa, 0, 27);
	xm_block_space_split(bsa, 0, 38);
	xm_block_space_split(bsa, 1, 5);
	xm_block_space_split(bsa, 1, 7);
	xm_block_space_split(bsa, 1, 27);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 2, 7);
	xm_block_space_split(bsa, 2, 18);
	xm_block_space_split(bsa, 2, 21);
	a = xm_tensor_create_canonical(bsa, type, allocator);
	fill_random(a);
	bsb = xm_block_space_permute_clone(bsa, permutation);
	assert(bsb);
	xm_block_space_free(bsa);
	b = xm_tensor_create_canonical(bsb, type, allocator);
	xm_block_space_free(bsb);
	xm_copy(b, s, a, "ijk", "kij");
	ia = xm_dim_zero(dims.n);
	while (xm_dim_ne(&ia, &dims)) {
		xm_scalar_t aa, bb;
		ib = xm_dim_permute(&ia, &permutation);
		aa = xm_tensor_get_element(a, ia);
		bb = xm_tensor_get_element(b, ib);
		if (!scalar_eq(aa*s, bb, type))
			fatal("tensors do not match");
		xm_dim_inc(&ia, &dims);
	}
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_copy_4(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b;
	xm_dim_t idx, absdims;

	allocator = xm_allocator_create(path);
	bsa = xm_block_space_create(xm_dim_3(2, 9, 11));
	xm_block_space_split(bsa, 0, 1);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 1, 4);
	xm_block_space_split(bsa, 2, 8);
	xm_block_space_split(bsa, 2, 4);
	xm_block_space_split(bsa, 2, 3);
	bsb = xm_block_space_permute_clone(bsa, xm_dim_3(2, 0, 1));
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create_canonical(bsb, type, allocator);
	xm_block_space_free(bsa);
	xm_block_space_free(bsb);

	xm_tensor_set_canonical_block(a, xm_dim_3(0, 0, 0));
	xm_tensor_set_derivative_block(a, xm_dim_3(1, 1, 3), xm_dim_3(0, 0, 0),
	    xm_dim_3(0, 1, 2), -7);
	xm_tensor_set_canonical_block(a, xm_dim_3(0, 1, 1));
	xm_tensor_set_canonical_block(a, xm_dim_3(1, 2, 0));
	xm_tensor_set_canonical_block(a, xm_dim_3(1, 2, 2));

	xm_set(a, 0);
	fill_random(b);
	xm_copy(b, random_scalar(type), a, "ijk", "kij");

	absdims = xm_tensor_get_abs_dims(b);
	idx = xm_dim_zero(absdims.n);
	while (xm_dim_ne(&idx, &absdims)) {
		xm_scalar_t e = xm_tensor_get_element(b, idx);
		if (!scalar_eq(e, 0, type))
			fatal("all elements must be zero");
		xm_dim_inc(&idx, &absdims);
	}

	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_copy_5(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b;

	allocator = xm_allocator_create(path);
	bsa = xm_block_space_create(xm_dim_4(2, 9, 11, 3));
	xm_block_space_split(bsa, 0, 1);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 1, 4);
	xm_block_space_split(bsa, 2, 8);
	xm_block_space_split(bsa, 2, 4);
	xm_block_space_split(bsa, 2, 3);
	a = xm_tensor_create(bsa, type, allocator);
	b = xm_tensor_create_canonical(bsa, type, allocator);
	xm_block_space_free(bsa);

	xm_tensor_set_canonical_block(a, xm_dim_4(0, 0, 0, 0));
	xm_tensor_set_derivative_block(a, xm_dim_4(1, 1, 3, 0),
	    xm_dim_4(0, 0, 0, 0), xm_dim_4(0, 1, 2, 3), -3);
	xm_tensor_set_canonical_block(a, xm_dim_4(0, 1, 1, 0));
	xm_tensor_set_canonical_block(a, xm_dim_4(1, 2, 0, 0));
	xm_tensor_set_canonical_block(a, xm_dim_4(1, 2, 2, 0));

	fill_random(a);
	fill_random(b);
	xm_copy(b, 1, a, "abcd", "abcd");
	compare_tensors(a, b);

	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_copy_6(const char *path, xm_scalar_type_t typea)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b;
	xm_scalar_type_t typeb = typea;

	if (typea == XM_SCALAR_FLOAT)
		typeb = XM_SCALAR_DOUBLE;
	if (typea == XM_SCALAR_FLOAT_COMPLEX)
		typeb = XM_SCALAR_DOUBLE_COMPLEX;
	allocator = xm_allocator_create(path);
	bsa = xm_block_space_create(xm_dim_4(2, 9, 11, 3));
	xm_block_space_split(bsa, 0, 1);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 1, 4);
	xm_block_space_split(bsa, 2, 8);
	xm_block_space_split(bsa, 2, 4);
	xm_block_space_split(bsa, 2, 3);
	a = xm_tensor_create(bsa, typea, allocator);
	b = xm_tensor_create_canonical(bsa, typeb, allocator);
	xm_block_space_free(bsa);

	xm_tensor_set_canonical_block(a, xm_dim_4(0, 0, 0, 0));
	xm_tensor_set_derivative_block(a, xm_dim_4(1, 1, 3, 0),
	    xm_dim_4(0, 0, 0, 0), xm_dim_4(0, 1, 2, 3), -3);
	xm_tensor_set_canonical_block(a, xm_dim_4(0, 1, 1, 0));
	xm_tensor_set_canonical_block(a, xm_dim_4(1, 2, 0, 0));
	xm_tensor_set_canonical_block(a, xm_dim_4(1, 2, 2, 0));

	fill_random(a);
	fill_random(b);
	xm_copy(b, 1, a, "abcd", "abcd");
	compare_tensors(a, b);

	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_allocator_destroy(allocator);
}

static void
test_dim(void)
{
	xm_dim_t idx1, idx2, dim;
	size_t i, offset;

	dim = xm_dim_4(5, 6, 3, 1);
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		idx2 = xm_dim_from_offset(offset, &dim);
		if (xm_dim_ne(&idx1, &idx2))
			fatal("dims do not match");
		xm_dim_inc(&idx1, &dim);
	}

	dim = xm_dim_3(1, 31, 16);
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		idx2 = xm_dim_from_offset(offset, &dim);
		if (xm_dim_ne(&idx1, &idx2))
			fatal("dims do not match");
		xm_dim_inc(&idx1, &dim);
	}

	i = 0;
	dim = xm_dim_8(5, 2, 3, 2, 1, 2, 5, 4);
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		if (offset != i)
			fatal("dims are not sequential");
		xm_dim_inc(&idx1, &dim);
		i++;
	}
}

static void
test_blockspace(void)
{
	xm_block_space_t *bsa, *bsb;
	xm_dim_t dima, dimb, nblocks, idx;

	bsa = xm_block_space_create(xm_dim_8(1,2,3,4,5,6,7,8));
	xm_block_space_autosplit(bsa);
	dima = xm_block_space_get_nblocks(bsa);
	dimb = xm_dim_same(8, 1);
	assert(xm_dim_eq(&dima, &dimb));
	dima = xm_block_space_get_block_dims(bsa, xm_dim_zero(8));
	dimb = xm_dim_8(1,2,3,4,5,6,7,8);
	assert(xm_dim_eq(&dima, &dimb));
	xm_block_space_free(bsa);

	bsa = xm_block_space_create(xm_dim_8(10,20,30,40,50,60,70,80));
	xm_block_space_autosplit(bsa);
	nblocks = xm_block_space_get_nblocks(bsa);
	dima = xm_dim_8(1,1,1,2,2,2,3,3);
	assert(xm_dim_eq(&nblocks, &dima));
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		dima = xm_block_space_get_block_dims(bsa, idx);
		dimb = xm_dim_same(8, 31);
		assert(xm_dim_less(&dima, &dimb));
		xm_dim_inc(&idx, &nblocks);
	}
	xm_block_space_free(bsa);

	bsa = xm_block_space_create(xm_dim_5(30,31,32,33,34));
	xm_block_space_autosplit(bsa);
	nblocks = xm_block_space_get_nblocks(bsa);
	dima = xm_dim_5(1,1,1,2,2);
	assert(xm_dim_eq(&nblocks, &dima));
	dima = xm_block_space_get_block_dims(bsa, xm_dim_zero(5));
	dimb = xm_dim_5(30,31,32,16,18);
	assert(xm_dim_eq(&dima, &dimb));
	xm_block_space_free(bsa);

	bsa = xm_block_space_create(xm_dim_7(102,40,203,593,15,84,1934));
	bsb = xm_block_space_create(xm_dim_6(40,593,83,84,85,1934));
	assert(xm_block_space_eq1(bsa, 1, bsb, 0));
	assert(xm_block_space_eq1(bsa, 3, bsb, 1));
	assert(!xm_block_space_eq1(bsa, 5, bsb, 2));
	assert(xm_block_space_eq1(bsa, 5, bsb, 3));
	assert(!xm_block_space_eq1(bsa, 5, bsb, 4));
	assert(xm_block_space_eq1(bsa, 6, bsb, 5));
	xm_block_space_autosplit(bsa);
	xm_block_space_autosplit(bsb);
	assert(xm_block_space_eq1(bsa, 1, bsb, 0));
	assert(xm_block_space_eq1(bsa, 3, bsb, 1));
	assert(!xm_block_space_eq1(bsa, 5, bsb, 2));
	assert(xm_block_space_eq1(bsa, 5, bsb, 3));
	assert(!xm_block_space_eq1(bsa, 5, bsb, 4));
	assert(xm_block_space_eq1(bsa, 6, bsb, 5));
	xm_block_space_free(bsa);
	xm_block_space_free(bsb);
}

static void
test_set(const char *path, xm_scalar_type_t type)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bs;
	xm_tensor_t *a;
	xm_dim_t dims, idx;
	const xm_scalar_t x = random_scalar(type);

	allocator = xm_allocator_create(path);
	assert(allocator);
	dims = xm_dim_8(3, 5, 1, 6, 3, 6, 9, 4);
	bs = xm_block_space_create(dims);
	assert(bs);
	xm_block_space_split(bs, 0, 1);
	xm_block_space_split(bs, 0, 1); /* test same */
	xm_block_space_split(bs, 1, 1);
	xm_block_space_split(bs, 1, 2);
	xm_block_space_split(bs, 3, 3);
	xm_block_space_split(bs, 5, 4);
	xm_block_space_split(bs, 6, 3);
	xm_block_space_split(bs, 6, 7);
	a = xm_tensor_create_canonical(bs, type, allocator);
	xm_block_space_free(bs);
	xm_set(a, x);
	idx = xm_dim_zero(dims.n);
	while (xm_dim_ne(&idx, &dims)) {
		xm_scalar_t aa = xm_tensor_get_element(a, idx);
		if (!scalar_eq(aa, x, type))
			fatal("elements are not equal");
		xm_dim_inc(&idx, &dims);
	}
	xm_tensor_free_block_data(a);
	xm_tensor_free(a);
	xm_allocator_destroy(allocator);
}

static const test_fn unfold_tests[] = {
	test_unfold_1,
	test_unfold_2,
	test_unfold_3,
};

static const test_fn copy_tests[] = {
	test_copy_1,
	test_copy_2,
	test_copy_3,
	test_copy_4,
	test_copy_5,
	test_copy_6,
};

static const struct two_tensor_test add_tests[] = {
	{ make_ab_1, "abcdefgh", "abcdefgh" },
	{ make_ab_2, "ijkl", "ijkl" },
	{ make_ab_2, "ijkl", "ijlk" },
	{ make_ab_2, "ijkl", "jkil" },
	{ make_ab_2, "ijkl", "lkji" },
	{ make_ab_3, "ij", "ij" },
	{ make_ab_3, "ji", "ij" },
	{ make_ab_4, "ij", "ij" },
	{ make_ab_4, "ij", "ji" },
};

static const struct two_tensor_test mul_div_tests[] = {
	{ make_ab_1, "abcdefgh", "abcdefgh" },
	{ make_ab_2, "ijkl", "ijkl" },
	{ make_ab_2, "ijkl", "ijlk" },
	{ make_ab_2, "ijkl", "jkil" },
	{ make_ab_2, "ijkl", "lkji" },
	{ make_ab_5, "ij", "ij" },
};

static const struct two_tensor_test dot_tests[] = {
	{ make_ab_1, "abcdefgh", "abcdefgh" },
	{ make_ab_2, "ijkl", "ijkl" },
	{ make_ab_2, "ijkl", "ijlk" },
	{ make_ab_2, "ijkl", "jkil" },
	{ make_ab_2, "ijkl", "lkji" },
	{ make_ab_3, "ij", "ij" },
	{ make_ab_3, "ji", "ij" },
	{ make_ab_4, "ij", "ij" },
	{ make_ab_4, "ij", "ji" },
	{ make_ab_5, "ij", "ij" },
};

static const struct contract_test contract_tests[] = {
	{ make_abc_1, "ik", "kj", "ij" },
	{ make_abc_1, "ik", "kj", "ji" },
	{ make_abc_1, "ik", "jk", "ij" },
	{ make_abc_1, "ik", "jk", "ji" },
	{ make_abc_1, "ki", "kj", "ij" },
	{ make_abc_1, "ki", "kj", "ji" },
	{ make_abc_1, "ki", "jk", "ij" },
	{ make_abc_1, "ki", "jk", "ji" },
	{ make_abc_2, "ik", "kj", "ij" },
	{ make_abc_2, "ik", "kj", "ji" },
	{ make_abc_2, "ik", "jk", "ij" },
	{ make_abc_2, "ik", "jk", "ji" },
	{ make_abc_2, "ki", "kj", "ij" },
	{ make_abc_2, "ki", "kj", "ji" },
	{ make_abc_2, "ki", "jk", "ij" },
	{ make_abc_2, "ki", "jk", "ji" },
	{ make_abc_3, "abcdefgh", "abcdefgi", "ih" },
	{ make_abc_3, "abcdefgi", "abcdefgh", "ih" },
	{ make_abc_3, "abcdxfgh", "abcdyfgh", "xy" },
	{ make_abc_3, "abcdefgh", "obcdefgh", "ao" },
	{ make_abc_3, "abcdefgh", "obcdefgh", "oa" },
	{ make_abc_4, "abcdef", "abcijk", "ijkdef" },
	{ make_abc_4, "abcdef", "aibjck", "ijkdef" },
	{ make_abc_4, "badcfe", "xyzdef", "xyzabc" },
	{ make_abc_5, "ik", "kj", "ij" },
	{ make_abc_5, "ik", "kj", "ji" },
	{ make_abc_5, "ik", "jk", "ij" },
	{ make_abc_5, "ik", "jk", "ji" },
	{ make_abc_5, "ki", "kj", "ij" },
	{ make_abc_5, "ki", "kj", "ji" },
	{ make_abc_5, "ki", "jk", "ij" },
	{ make_abc_5, "ki", "jk", "ji" },
	{ make_abc_6, "y", "xz", "xyz" },
	{ make_abc_7, "abc", "abd", "dc" },
	{ make_abc_8, "ab", "b", "a" },
	{ make_abc_9, "ijab", "abcd", "ijcd" },
	{ make_abc_9, "ijcd", "abcd", "ijab" },
	{ make_abc_10, "abcd", "ijab", "ijcd" },
	{ make_abc_10, "cdab", "ijab", "ijcd" },
	{ make_abc_11, "abcd", "ijcd", "ijab" },
	{ make_abc_12, "abcf", "abce", "ef" },
	{ make_abc_12, "abfc", "abce", "ef" },
	{ make_abc_12, "fabc", "bace", "ef" },
	{ make_abc_12, "abcf", "acbe", "fe" },
	{ make_abc_12, "afbc", "eabc", "fe" },
	{ make_abc_12, "afcb", "bace", "fe" },
};

static void
run_tests(const char *path, xm_scalar_type_t type)
{
	size_t i;

	printf("dim test 1... ");
	fflush(stdout);
	test_dim();
	printf("success\n");

	printf("block-space test 1... ");
	fflush(stdout);
	test_blockspace();
	printf("success\n");

	printf("set test 1... ");
	fflush(stdout);
	test_set(path, type);
	printf("success\n");

	for (i = 0; i < sizeof unfold_tests / sizeof *unfold_tests; i++) {
		printf("unfold test %zu... ", i+1);
		fflush(stdout);
		unfold_tests[i](path, type);
		printf("success\n");
	}
	for (i = 0; i < sizeof copy_tests / sizeof *copy_tests; i++) {
		printf("copy test %zu... ", i+1);
		fflush(stdout);
		copy_tests[i](path, type);
		printf("success\n");
	}
	for (i = 0; i < sizeof add_tests / sizeof *add_tests; i++) {
		printf("add test %zu... ", i+1);
		fflush(stdout);
		test_add(&add_tests[i], path, type, 0, 0);
		test_add(&add_tests[i], path, type, 0, random_scalar(type));
		test_add(&add_tests[i], path, type, random_scalar(type), 0);
		test_add(&add_tests[i], path, type, random_scalar(type),
		    random_scalar(type));
		printf("success\n");
	}
	for (i = 0; i < sizeof mul_div_tests / sizeof *mul_div_tests; i++) {
		printf("mul test %zu... ", i+1);
		fflush(stdout);
		test_mul(&mul_div_tests[i], path, type);
		printf("success\n");
	}
	for (i = 0; i < sizeof mul_div_tests / sizeof *mul_div_tests; i++) {
		printf("div test %zu... ", i+1);
		fflush(stdout);
		test_div(&mul_div_tests[i], path, type);
		printf("success\n");
	}
	for (i = 0; i < sizeof dot_tests / sizeof *dot_tests; i++) {
		printf("dot test %zu... ", i+1);
		fflush(stdout);
		test_dot(&dot_tests[i], path, type);
		printf("success\n");
	}
	for (i = 0; i < sizeof contract_tests / sizeof *contract_tests; i++) {
		printf("contract test %2zu... ", i+1);
		fflush(stdout);
		test_contract(&contract_tests[i], path, type, 0, 0);
		test_contract(&contract_tests[i], path, type,
		    0, random_scalar(type));
		test_contract(&contract_tests[i], path, type,
		    random_scalar(type), 0);
		test_contract(&contract_tests[i], path, type,
		    random_scalar(type), random_scalar(type));
		printf("success\n");
	}
}

int
main(int argc, char **argv)
{
	const char *path = "xmpagefile";

#ifdef XM_USE_MPI
	MPI_Init(&argc, &argv);
	srand48(0);
	puts("Testing float...");
	run_tests(path, XM_SCALAR_FLOAT);
	puts("Testing double...");
	run_tests(path, XM_SCALAR_DOUBLE);
	puts("Testing float complex...");
	run_tests(path, XM_SCALAR_FLOAT_COMPLEX);
	puts("Testing double complex...");
	run_tests(path, XM_SCALAR_DOUBLE_COMPLEX);
	MPI_Finalize();
#else
	(void)argc;
	(void)argv;
	puts("Testing float...");
	run_tests(NULL, XM_SCALAR_FLOAT);
	puts("Testing double...");
	run_tests(NULL, XM_SCALAR_DOUBLE);
	puts("Testing float complex...");
	run_tests(NULL, XM_SCALAR_FLOAT_COMPLEX);
	puts("Testing double complex...");
	run_tests(NULL, XM_SCALAR_DOUBLE_COMPLEX);
	puts("Testing float...");
	run_tests(path, XM_SCALAR_FLOAT);
	puts("Testing double...");
	run_tests(path, XM_SCALAR_DOUBLE);
	puts("Testing float complex...");
	run_tests(path, XM_SCALAR_FLOAT_COMPLEX);
	puts("Testing double complex...");
	run_tests(path, XM_SCALAR_DOUBLE_COMPLEX);
#endif
	return 0;
}
