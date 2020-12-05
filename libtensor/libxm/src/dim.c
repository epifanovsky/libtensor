/*
 * Copyright (c) 2014-2017 Ilya Kaliman
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

#include "dim.h"

#include <assert.h>

xm_dim_t
xm_dim_zero(size_t ndim)
{
	return (xm_dim_same(ndim, 0));
}

xm_dim_t
xm_dim_same(size_t ndim, size_t dim)
{
	xm_dim_t ret;

	assert(ndim <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < ndim; ret.n++)
		ret.i[ret.n] = dim;
	return (ret);
}

xm_dim_t
xm_dim_1(size_t dim1)
{
	xm_dim_t dim;

	dim.n = 1;
	dim.i[0] = dim1;

	return (dim);
}

xm_dim_t
xm_dim_2(size_t dim1, size_t dim2)
{
	xm_dim_t dim;

	dim.n = 2;
	dim.i[0] = dim1;
	dim.i[1] = dim2;

	return (dim);
}

xm_dim_t
xm_dim_3(size_t dim1, size_t dim2, size_t dim3)
{
	xm_dim_t dim;

	dim.n = 3;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;

	return (dim);
}

xm_dim_t
xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4)
{
	xm_dim_t dim;

	dim.n = 4;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;

	return (dim);
}

xm_dim_t
xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5)
{
	xm_dim_t dim;

	dim.n = 5;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;

	return (dim);
}

xm_dim_t
xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6)
{
	xm_dim_t dim;

	dim.n = 6;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;
	dim.i[5] = dim6;

	return (dim);
}

xm_dim_t
xm_dim_7(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7)
{
	xm_dim_t dim;

	dim.n = 7;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;
	dim.i[5] = dim6;
	dim.i[6] = dim7;

	return (dim);
}

xm_dim_t
xm_dim_8(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7, size_t dim8)
{
	xm_dim_t dim;

	dim.n = 8;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;
	dim.i[5] = dim6;
	dim.i[6] = dim7;
	dim.i[7] = dim8;

	return (dim);
}

int
xm_dim_eq(const xm_dim_t *a, const xm_dim_t *b)
{
	size_t i;

	if (a->n != b->n)
		return (0);
	for (i = 0; i < a->n; i++)
		if (a->i[i] != b->i[i])
			return (0);
	return (1);
}

int
xm_dim_ne(const xm_dim_t *a, const xm_dim_t *b)
{
	return (!xm_dim_eq(a, b));
}

int
xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t i;

	assert(idx->n == dim->n);

	for (i = 0; i < idx->n; i++)
		if (idx->i[i] >= dim->i[i])
			return (0);
	return (1);
}

size_t
xm_dim_dot(const xm_dim_t *dim)
{
	size_t i, ret = 1;

	for (i = 0; i < dim->n; i++)
		ret *= dim->i[i];
	return (ret);
}

size_t
xm_dim_offset(const xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t ret = 0;

	assert(xm_dim_less(idx, dim));

	switch (idx->n) {
	case 8: ret += idx->i[7] * dim->i[6] * dim->i[5] * dim->i[4] *
		       dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	/* FALLTHRU */
	case 7: ret += idx->i[6] * dim->i[5] * dim->i[4] * dim->i[3] *
		       dim->i[2] * dim->i[1] * dim->i[0];
	/* FALLTHRU */
	case 6: ret += idx->i[5] * dim->i[4] * dim->i[3] * dim->i[2] *
		       dim->i[1] * dim->i[0];
	/* FALLTHRU */
	case 5: ret += idx->i[4] * dim->i[3] * dim->i[2] * dim->i[1] *
		       dim->i[0];
	/* FALLTHRU */
	case 4: ret += idx->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	/* FALLTHRU */
	case 3: ret += idx->i[2] * dim->i[1] * dim->i[0];
	/* FALLTHRU */
	case 2: ret += idx->i[1] * dim->i[0];
	/* FALLTHRU */
	case 1: ret += idx->i[0];
	}
	return (ret);
}

xm_dim_t
xm_dim_from_offset(size_t offset, const xm_dim_t *dim)
{
	xm_dim_t ret;
	size_t p;

	ret.n = dim->n;

	switch (ret.n) {
	case 8: p = dim->i[6] * dim->i[5] * dim->i[4] * dim->i[3] *
		    dim->i[2] * dim->i[1] * dim->i[0];
		ret.i[7] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 7: p = dim->i[5] * dim->i[4] * dim->i[3] * dim->i[2] *
		    dim->i[1] * dim->i[0];
		ret.i[6] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 6: p = dim->i[4] * dim->i[3] * dim->i[2] * dim->i[1] *
		    dim->i[0];
		ret.i[5] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 5: p = dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
		ret.i[4] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 4: p = dim->i[2] * dim->i[1] * dim->i[0];
		ret.i[3] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 3: p = dim->i[1] * dim->i[0];
		ret.i[2] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 2: p = dim->i[0];
		ret.i[1] = offset / p;
		offset %= p;
	/* FALLTHRU */
	case 1: ret.i[0] = offset;
	}
	return (ret);
}

void
xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < idx->n; i++) {
		idx->i[i] += carry;
		carry = idx->i[i] / dim->i[i];
		idx->i[i] %= dim->i[i];
	}
	if (carry)
		*idx = *dim;
}

void
xm_dim_zero_mask(xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i;

	for (i = 0; i < mask->n; i++)
		dim->i[mask->i[i]] = 0;
}

void
xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *maska, const xm_dim_t *b,
    const xm_dim_t *maskb)
{
	size_t i;

	assert(maska->n == maskb->n);

	for (i = 0; i < maska->n; i++)
		a->i[maska->i[i]] = b->i[maskb->i[i]];
}

size_t
xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, ret = 1;

	for (i = 0; i < mask->n; i++)
		ret *= dim->i[mask->i[i]];
	return (ret);
}

void
xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < mask->n; i++) {
		idx->i[mask->i[i]] += carry;
		carry = idx->i[mask->i[i]] / dim->i[mask->i[i]];
		idx->i[mask->i[i]] %= dim->i[mask->i[i]];
	}
}

xm_dim_t
xm_dim_identity_permutation(size_t ndim)
{
	xm_dim_t ret;

	assert(ndim <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < ndim; ret.n++)
		ret.i[ret.n] = ret.n;
	return (ret);
}

xm_dim_t
xm_dim_permute(const xm_dim_t *idx, const xm_dim_t *permutation)
{
	xm_dim_t ret;

	assert(idx->n == permutation->n);

	ret.n = idx->n;
	switch (ret.n) {
	case 8: ret.i[permutation->i[7]] = idx->i[7]; /* FALLTHRU */
	case 7: ret.i[permutation->i[6]] = idx->i[6]; /* FALLTHRU */
	case 6: ret.i[permutation->i[5]] = idx->i[5]; /* FALLTHRU */
	case 5: ret.i[permutation->i[4]] = idx->i[4]; /* FALLTHRU */
	case 4: ret.i[permutation->i[3]] = idx->i[3]; /* FALLTHRU */
	case 3: ret.i[permutation->i[2]] = idx->i[2]; /* FALLTHRU */
	case 2: ret.i[permutation->i[1]] = idx->i[1]; /* FALLTHRU */
	case 1: ret.i[permutation->i[0]] = idx->i[0];
	}
	return (ret);
}
