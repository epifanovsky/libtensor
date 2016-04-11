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

typedef struct test (*make_test_fn_t)(void);
typedef int (*init_fn_t)(struct xm_tensor *, struct xm_allocator *,
    size_t, int);

struct test {
	size_t block_size;
	xm_dim_t dima;
	xm_dim_t dimb;
	xm_dim_t dimc;
	xm_dim_t pdima;
	xm_dim_t pdimb;
	xm_dim_t pdimc;
	const char *idxa;
	const char *idxb;
	const char *idxc;
	init_fn_t init_a;
	init_fn_t init_b;
	init_fn_t init_c;
	xm_scalar_t alpha;
	xm_scalar_t beta;
	void (*ref_compare)(struct xm_tensor *,
			    struct xm_tensor *,
			    struct xm_tensor *,
			    struct xm_tensor *,
			    xm_scalar_t, xm_scalar_t);
};

static void
fatal(const char *msg)
{
	fprintf(stderr, "fatal error: %s\n", msg);
	abort();
}

static size_t
rnd(size_t from, size_t to)
{
	size_t x;

#ifdef HAVE_ARC4RANDOM
	x = from + arc4random_uniform((uint32_t)(to - from + 1));
#else
	x = from + ((size_t)(rand())) % (to - from + 1);
#endif

	return (x);
}

static void
ref_compare_1(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
			idxa.i[0] = idxc.i[1];
			idxa.i[1] = i;
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[0];
			idxb.i[1] = i;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} }
}

static void
ref_compare_3(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
			idxa.i[0] = idxc.i[1];
			idxa.i[1] = i;
			idxa.i[2] = idxc.i[3];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[2];
			idxb.i[1] = i;
			idxb.i[2] = idxc.i[0];
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } }
}

static void
ref_compare_4(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i, j;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
		for (j = 0; j < dima.i[2]; j++) {
			idxa.i[0] = idxc.i[0];
			idxa.i[1] = i;
			idxa.i[2] = j;
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[1];
			idxb.i[1] = i;
			idxb.i[2] = j;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		} }
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} }
}

static void
ref_compare_5(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i, j;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
		for (j = 0; j < dima.i[3]; j++) {
			idxa.i[0] = idxc.i[2];
			idxa.i[1] = i;
			idxa.i[2] = idxc.i[3];
			idxa.i[3] = j;
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[0];
			idxb.i[1] = i;
			idxb.i[2] = idxc.i[1];
			idxb.i[3] = j;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		} }
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } }
}

static void
ref_compare_7(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i, j;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[2]; i++) {
		for (j = 0; j < dima.i[3]; j++) {
			idxa.i[0] = idxc.i[2];
			idxa.i[1] = idxc.i[3];
			idxa.i[2] = i;
			idxa.i[3] = j;
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[0];
			idxb.i[1] = idxc.i[1];
			idxb.i[2] = i;
			idxb.i[3] = j;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		} }
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } }
}

static void
ref_compare_9(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
			idxa.i[0] = idxc.i[1];
			idxa.i[1] = i;
			idxa.i[2] = idxc.i[0];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[2];
			idxb.i[1] = i;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } }
}

static void
ref_compare_10(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[0]; i++) {
			idxa.i[0] = i;
			idxa.i[1] = idxc.i[1];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[0];
			idxb.i[1] = idxc.i[2];
			idxb.i[2] = i;
			idxb.i[3] = idxc.i[3];
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } }
}

static void
ref_compare_11(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
	for (idxc.i[4] = 0; idxc.i[4] < dimc.i[4]; idxc.i[4]++) {
	for (idxc.i[5] = 0; idxc.i[5] < dimc.i[5]; idxc.i[5]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[1]; i++) {
			idxa.i[0] = idxc.i[0];
			idxa.i[1] = i;
			idxa.i[2] = idxc.i[1];
			idxa.i[3] = idxc.i[2];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[3];
			idxb.i[1] = idxc.i[4];
			idxb.i[2] = i;
			idxb.i[3] = idxc.i[5];
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } } } }
}

static void
ref_compare_12(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
	for (idxc.i[4] = 0; idxc.i[4] < dimc.i[4]; idxc.i[4]++) {
	for (idxc.i[5] = 0; idxc.i[5] < dimc.i[5]; idxc.i[5]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[2]; i++) {
			idxa.i[0] = idxc.i[0];
			idxa.i[1] = idxc.i[1];
			idxa.i[2] = i;
			idxa.i[3] = idxc.i[3];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[2];
			idxb.i[1] = i;
			idxb.i[2] = idxc.i[4];
			idxb.i[3] = idxc.i[5];
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } } } }
}

static void
ref_compare_13(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[0]; i++) {
			idxa.i[0] = i;
			idxa.i[1] = idxc.i[1];
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = i;
			idxb.i[1] = idxc.i[0];
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		}
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} }
}

static void
ref_compare_15(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	size_t i, j;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		for (i = 0; i < dima.i[0]; i++) {
		for (j = 0; j < dima.i[1]; j++) {
			idxa.i[0] = i;
			idxa.i[1] = j;
			aa = xm_tensor_get_abs_element(a, &idxa);
			idxb.i[0] = idxc.i[0];
			idxb.i[1] = idxc.i[1];
			idxb.i[2] = i;
			idxb.i[3] = j;
			bb = xm_tensor_get_abs_element(b, &idxb);
			ref += alpha * aa * bb;
		} }
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} }
}

static void
ref_compare_16(struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, struct xm_tensor *d, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_dim_t dima, dimb, dimc, idxa, idxb, idxc;
	xm_scalar_t aa, bb, dd, ref;

	dima = xm_tensor_get_abs_dim(a);
	dimb = xm_tensor_get_abs_dim(b);
	dimc = xm_tensor_get_abs_dim(c);
	idxa.n = dima.n;
	idxb.n = dimb.n;
	idxc.n = dimc.n;

	for (idxc.i[0] = 0; idxc.i[0] < dimc.i[0]; idxc.i[0]++) {
	for (idxc.i[1] = 0; idxc.i[1] < dimc.i[1]; idxc.i[1]++) {
	for (idxc.i[2] = 0; idxc.i[2] < dimc.i[2]; idxc.i[2]++) {
	for (idxc.i[3] = 0; idxc.i[3] < dimc.i[3]; idxc.i[3]++) {
		ref = beta * xm_tensor_get_abs_element(c, &idxc);
		idxa.i[0] = idxc.i[0];
		idxa.i[1] = idxc.i[1];
		aa = xm_tensor_get_abs_element(a, &idxa);
		idxb.i[0] = idxc.i[2];
		idxb.i[1] = idxc.i[3];
		bb = xm_tensor_get_abs_element(b, &idxb);
		ref += alpha * aa * bb;
		dd = xm_tensor_get_abs_element(d, &idxc);
		if (xm_abs(ref - dd) > EPSILON)
			fatal("result != reference");
	} } } }
}

static struct test
make_test_1(void)
{
	const size_t max_block_size = 9;
	const size_t max_dim = 30;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_2(rnd(1, max_dim), rnd(1, max_dim));
	t.dimb = xm_dim_2(rnd(1, max_dim), t.dima.i[1]);
	t.dimc = xm_dim_2(t.dimb.i[0], t.dima.i[0]);
	t.pdima = xm_dim_same(2, pdim);
	t.pdimb = xm_dim_same(2, pdim);
	t.pdimc = xm_dim_same(2, pdim);
	t.idxa = "ab";
	t.idxb = "cb";
	t.idxc = "ca";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_1;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_2(void)
{
	const size_t max_block_size = 9;
	const size_t max_dim = 20;
	const size_t pdim = rnd(1, 2);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_2(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim));
	t.dimb = xm_dim_2(t.dima.i[1], t.dima.i[1]);
	t.dimc = xm_dim_2(t.dimb.i[0], t.dima.i[0]);
	t.pdima = xm_dim_same(2, pdim);
	t.pdimb = xm_dim_same(2, pdim);
	t.pdimc = xm_dim_same(2, pdim);
	t.idxa = "ab";
	t.idxb = "cb";
	t.idxc = "ca";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init_oo;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_1;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_3(void)
{
	const size_t max_block_size = 5;
	const size_t max_dim = 4;
	const size_t pdim = rnd(1, 2);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_3(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim));
	t.dimb = xm_dim_3(pdim * rnd(1, max_dim),
			  t.dima.i[1],
			  pdim * rnd(1, max_dim));
	t.dimc = xm_dim_4(t.dimb.i[2],
			  t.dima.i[0],
			  t.dimb.i[0],
			  t.dima.i[2]);
	t.pdima = xm_dim_same(3, pdim);
	t.pdimb = xm_dim_same(3, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "abc";
	t.idxb = "dbe";
	t.idxc = "eadc";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_3;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_4(void)
{
	const size_t max_block_size = 6;
	const size_t max_dim = 6;
	const size_t pdim = rnd(1, 2);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_3(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim));
	t.dimb = xm_dim_3(pdim * rnd(1, max_dim),
			  t.dima.i[1],
			  t.dima.i[2]);
	t.dimc = xm_dim_2(t.dima.i[0],
			  t.dimb.i[0]);
	t.pdima = xm_dim_same(3, pdim);
	t.pdimb = xm_dim_same(3, pdim);
	t.pdimc = xm_dim_same(2, pdim);
	t.idxa = "abc";
	t.idxb = "dbc";
	t.idxc = "ad";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_4;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_5(void)
{
	const size_t max_block_size = 4;
	const size_t max_dim = 4;
	const size_t pdim = rnd(1, 2);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim));
	t.dimb = xm_dim_4(pdim * rnd(1, max_dim),
			  t.dima.i[1],
			  pdim * rnd(1, max_dim),
			  t.dima.i[3]);
	t.dimc = xm_dim_4(t.dimb.i[0],
			  t.dimb.i[2],
			  t.dima.i[0],
			  t.dima.i[2]);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "abcd";
	t.idxb = "ibjd";
	t.idxc = "ijac";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_5;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_6(void)
{
	const size_t max_block_size = 3;
	const size_t max_dim = 5;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(rnd(1, max_dim),
			  rnd(1, max_dim),
			  rnd(1, max_dim),
			  rnd(1, max_dim));
	t.dimb = xm_dim_4(t.dima.i[1],
			  t.dima.i[1],
			  t.dima.i[3],
			  t.dima.i[3]);
	t.dimc = xm_dim_4(t.dimb.i[0],
			  t.dimb.i[2],
			  t.dima.i[0],
			  t.dima.i[2]);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "abcd";
	t.idxb = "ibjd";
	t.idxc = "ijac";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init_oovv;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_5;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_7(void)
{
	const size_t max_block_size = 3;
	const size_t max_dim = 5;
	const size_t pdim = rnd(1, 2);
	const size_t o = pdim * rnd(1, max_dim);
	const size_t x = pdim * rnd(1, max_dim);
	const size_t v = pdim * rnd(1, max_dim);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(o, o, v, v);
	t.dimb = xm_dim_4(x, x, v, v);
	t.dimc = xm_dim_4(x, x, o, o);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "ijab";
	t.idxb = "klab";
	t.idxc = "klij";
	t.init_a = xm_tensor_init_oovv;
	t.init_b = xm_tensor_init_oovv;
	t.init_c = xm_tensor_init_oovv;
	t.ref_compare = ref_compare_7;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_8(void)
{
	const size_t max_block_size = 3;
	const size_t max_dim = 4;
	const size_t pdim = rnd(1, 2);
	const size_t o = pdim * rnd(1, max_dim);
	const size_t v = pdim * rnd(1, max_dim);
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(v, v, v, v);
	t.dimb = xm_dim_4(o, o, v, v);
	t.dimc = xm_dim_4(o, o, v, v);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "abcd";
	t.idxb = "ijcd";
	t.idxc = "ijab";
	t.init_a = xm_tensor_init_vvvv;
	t.init_b = xm_tensor_init_oovv;
	t.init_c = xm_tensor_init_oovv;
	t.ref_compare = ref_compare_7;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_9(void)
{
	const size_t max_block_size = 7;
	const size_t max_dim = 10;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_3(rnd(1, max_dim), rnd(1, max_dim), rnd(1, max_dim));
	t.dimb = xm_dim_2(rnd(1, max_dim), t.dima.i[1]);
	t.dimc = xm_dim_3(t.dima.i[2], t.dima.i[0], t.dimb.i[0]);
	t.pdima = xm_dim_same(3, pdim);
	t.pdimb = xm_dim_same(2, pdim);
	t.pdimc = xm_dim_same(3, pdim);
	t.idxa = "abc";
	t.idxb = "ib";
	t.idxc = "cai";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_9;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_10(void)
{
	const size_t max_block_size = 6;
	const size_t max_dim = 9;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_2(rnd(1, max_dim), rnd(1, max_dim));
	t.dimb = xm_dim_4(rnd(1, max_dim),
			  rnd(1, max_dim),
			  t.dima.i[0],
			  rnd(1, max_dim));
	t.dimc = xm_dim_4(t.dimb.i[0],
			  t.dima.i[1],
			  t.dimb.i[1],
			  t.dimb.i[3]);
	t.pdima = xm_dim_same(2, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "ab";
	t.idxb = "ijak";
	t.idxc = "ibjk";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_10;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_11(void)
{
	const size_t max_block_size = 3;
	const size_t max_dim = 4;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim));
	t.dimb = xm_dim_4(pdim * rnd(1, max_dim),
			  pdim * rnd(1, max_dim),
			  t.dima.i[1],
			  pdim * rnd(1, max_dim));
	t.dimc = xm_dim_6(t.dima.i[0],
			  t.dima.i[2],
			  t.dima.i[3],
			  t.dimb.i[0],
			  t.dimb.i[1],
			  t.dimb.i[3]);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(6, pdim);
	t.idxa = "idjk";
	t.idxb = "abdc";
	t.idxc = "ijkabc";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_11;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_12(void)
{
	const size_t max_block_size = 3;
	const size_t max_dim = 4;
	const size_t o = rnd(1, max_dim);
	const size_t v = rnd(1, max_dim);
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_4(o, o, v, v);
	t.dimb = xm_dim_4(o, v, v, v);
	t.dimc = xm_dim_6(o, o, o, v, v, v);
	t.pdima = xm_dim_same(4, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(6, pdim);
	t.idxa = "ijda";
	t.idxb = "kdbc";
	t.idxc = "ijkabc";
	t.init_a = xm_tensor_init_oovv;
	t.init_b = xm_tensor_init_ovvv;
	t.init_c = xm_tensor_init_ooovvv;
	t.ref_compare = ref_compare_12;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_13(void)
{
	struct test t;

	t.block_size = 0;
	t.dima = xm_dim_2(2, 2);
	t.dimb = xm_dim_2(2, 2);
	t.dimc = xm_dim_2(2, 2);
	t.pdima = xm_dim_same(2, 1);
	t.pdimb = xm_dim_same(2, 1);
	t.pdimc = xm_dim_same(2, 1);
	t.idxa = "ba";
	t.idxb = "bc";
	t.idxc = "ca";
	t.init_a = xm_tensor_init_13;
	t.init_b = xm_tensor_init_13;
	t.init_c = xm_tensor_init_13c;
	t.ref_compare = ref_compare_13;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_14(void)
{
	struct test t;

	t.block_size = 0;
	t.dima = xm_dim_4(2, 2, 2, 2);
	t.dimb = xm_dim_4(1, 1, 2, 2);
	t.dimc = xm_dim_4(1, 1, 2, 2);
	t.pdima = xm_dim_same(4, 1);
	t.pdimb = xm_dim_same(4, 1);
	t.pdimc = xm_dim_same(4, 1);
	t.idxa = "abcd";
	t.idxb = "ijcd";
	t.idxc = "ijab";
	t.init_a = xm_tensor_init_14;
	t.init_b = xm_tensor_init_14b;
	t.init_c = xm_tensor_init_14b;
	t.ref_compare = ref_compare_7;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_15(void)
{
	const size_t max_block_size = 6;
	const size_t max_dim = 9;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_2(rnd(1, max_dim), rnd(1, max_dim));
	t.dimb = xm_dim_4(rnd(1, max_dim),
			  rnd(1, max_dim),
			  t.dima.i[0],
			  t.dima.i[1]);
	t.dimc = xm_dim_2(t.dimb.i[0], t.dimb.i[1]);
	t.pdima = xm_dim_same(2, pdim);
	t.pdimb = xm_dim_same(4, pdim);
	t.pdimc = xm_dim_same(2, pdim);
	t.idxa = "ab";
	t.idxb = "ijab";
	t.idxc = "ij";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_15;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static struct test
make_test_16(void)
{
	const size_t max_block_size = 6;
	const size_t max_dim = 9;
	const size_t pdim = 1;
	struct test t;

	t.block_size = rnd(1, max_block_size);
	t.dima = xm_dim_2(rnd(1, max_dim), rnd(1, max_dim));
	t.dimb = xm_dim_2(rnd(1, max_dim), rnd(1, max_dim));
	t.dimc = xm_dim_4(t.dima.i[0], t.dima.i[1],
			  t.dimb.i[0], t.dimb.i[1]);
	t.pdima = xm_dim_same(2, pdim);
	t.pdimb = xm_dim_same(2, pdim);
	t.pdimc = xm_dim_same(4, pdim);
	t.idxa = "ab";
	t.idxb = "ij";
	t.idxc = "abij";
	t.init_a = xm_tensor_init;
	t.init_b = xm_tensor_init;
	t.init_c = xm_tensor_init;
	t.ref_compare = ref_compare_16;
	t.alpha = xm_random_scalar();
	t.beta = rnd(0, 5) ? xm_random_scalar() : 0.0;

	return (t);
}

static const make_test_fn_t tests[] = {
	make_test_1,
	make_test_2,
	make_test_3,
	make_test_4,
	make_test_5,
	make_test_6,
	make_test_7,
	make_test_8,
	make_test_9,
	make_test_10,
	make_test_11,
	make_test_12,
	make_test_13,
	make_test_14,
	make_test_15,
	make_test_16
};

static void
print_test_info(const struct test *test)
{
	printf("dima=%-2zudimb=%-2zudimc=%-2zubs=%-2zu",
	    test->dima.n, test->dimb.n, test->dimc.n,
	    test->block_size);
	fflush(stdout);
}

static void
run_test(int test_num, int skip)
{
	struct xm_allocator *allocator;
	struct xm_tensor *a, *b, *c, *d;
	struct test t;
	const char *path;
	size_t buf_bytes, id;
	int res;

	id = rnd(1, sizeof(tests) / sizeof(*tests));
	printf("test=%-4did=%-3zu", test_num, id);

	t = tests[id-1]();
	print_test_info(&t);

	path = rnd(0,1) ? NULL : "mapping";
	if ((allocator = xm_allocator_create(path)) == NULL)
		fatal("xm_allocator_create");

	if ((a = xm_tensor_create(allocator, &t.dima, "a")) == NULL)
		fatal("xm_tensor_create(a)");
	if ((b = xm_tensor_create(allocator, &t.dimb, "b")) == NULL)
		fatal("xm_tensor_create(b)");
	if ((c = xm_tensor_create(allocator, &t.dimc, "c")) == NULL)
		fatal("xm_tensor_create(c)");
	if ((d = xm_tensor_create(allocator, &t.dimc, "d")) == NULL)
		fatal("xm_tensor_create(d)");

	xm_tensor_set_part_dim(a, &t.pdima);
	xm_tensor_set_part_dim(b, &t.pdimb);
	xm_tensor_set_part_dim(c, &t.pdimc);
	xm_tensor_set_part_dim(d, &t.pdimc);

	if (t.init_a(a, allocator, t.block_size, XM_INIT_RAND))
		fatal("tensor_init(a)");
	if (t.init_b(b, allocator, t.block_size, XM_INIT_RAND))
		fatal("tensor_init(b)");
	if (t.init_c(c, allocator, t.block_size, XM_INIT_RAND))
		fatal("tensor_init(c)");
	if (t.init_c(d, allocator, t.block_size, XM_INIT_NONE))
		fatal("tensor_init(d)");
	xm_tensor_copy_data(d, c);

	buf_bytes = rnd(10000, 100000);
	xm_set_memory_limit(buf_bytes);

	if (!skip) {
		res = xm_contract(t.alpha, a, b, t.beta, d,
		    t.idxa, t.idxb, t.idxc);
		switch (res) {
		case XM_RESULT_SUCCESS:
			t.ref_compare(a, b, c, d, t.alpha, t.beta);
			printf("   success\n");
			break;
		case XM_RESULT_BUFFER_TOO_SMALL:
			printf("   buffer too small\n");
			break;
		case XM_RESULT_NO_MEMORY:
			fatal("xm_contract: out of memory");
		default:
			fatal("xm_contract: unknown error");
		}
	} else
		printf("   skipping\n");

	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_tensor_free(d);
	xm_allocator_destroy(allocator);
}

int
main(int argc, char **argv)
{
	int i, count, skip = 0;

	if (argc < 2) {
		fprintf(stderr, "usage: test count\n");
		return (1);
	}

	xm_set_log_stream(stderr);

	if ((count = atoi(argv[1])) < 0) {
		count = -count;
		skip = 1;
	}

	for (i = 1; i <= count; i++)
		run_test(i, i < count && skip);

	return (0);
}
