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

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <unistd.h>

#ifdef HAVE_BITSTRING_H
#include <bitstring.h>
#else
#include "compat/bitstring.h"
#endif

#include "xm.h"

struct xm_block {
	uintptr_t               data_ptr;
	xm_dim_t                idx;
	xm_dim_t                dim;
	xm_dim_t                permutation;
	xm_dim_t                source_idx;
	xm_scalar_t             scalar;
	int                     is_source;
	int                     is_nonzero;
	int                     is_initialized;
};

struct xm_tensor {
	char                   *label;
	xm_dim_t                dim;
	xm_dim_t                pdim;
	xm_dim_t                pdim0;
	xm_dim_t                pidx;
	struct xm_allocator    *allocator;
	struct xm_block        *blocks;
	xm_scalar_t            *block_buf;
	size_t                  block_buf_bytes;
};

struct async {
	struct xm_tensor       *tensor;
	xm_dim_t                blk_idx;
	xm_dim_t                mask_i;
	xm_dim_t                mask_j;
	size_t                  nblk_i;
	size_t                  nblk_j;
	bitstr_t               *skip_i;
	bitstr_t               *skip_j;
	xm_scalar_t            *data;
	size_t                  stride;
	xm_scalar_t            *from;
};

#if defined(XM_SCALAR_FLOAT)
#define xm_blas_gemm sgemm_
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#define xm_blas_gemm zgemm_
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#define xm_blas_gemm cgemm_
#else /* assume double */
#define xm_blas_gemm dgemm_
#endif

void xm_blas_gemm(char *, char *, int *, int *, int *, xm_scalar_t *,
    xm_scalar_t *, int *, xm_scalar_t *, int *, xm_scalar_t *,
    xm_scalar_t *, int *);

static void
gemm_wrapper(char transa, char transb, int m, int n, int k, xm_scalar_t alpha,
    xm_scalar_t *a, int lda, xm_scalar_t *b, int ldb, xm_scalar_t beta,
    xm_scalar_t *c, int ldc)
{
	xm_blas_gemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda,
	    b, &ldb, &beta, c, &ldc);
}

#define xm_log_line(msg) \
    xm_log("%s (in %s on line %d)", msg, __func__, __LINE__)
#define xm_log_perror(msg) \
    xm_log("%s (%s)", msg, strerror(errno))

/* stream to log to */
static FILE *xm_log_stream = NULL;

void
xm_set_log_stream(FILE *stream)
{
	xm_log_stream = stream;
}

static size_t
get_total_physmem(void)
{
	long pagesize, npages;

	pagesize = sysconf(_SC_PAGESIZE);
	npages = sysconf(_SC_PHYS_PAGES);

	return ((size_t)pagesize * (size_t)npages);
}

static size_t
get_buffer_size(void)
{
	return (get_total_physmem() / 2);
}

static void
xm_log(const char *fmt, ...)
{
	char buf[1024];
	va_list ap;

	if (xm_log_stream == NULL)
		return;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	fprintf(xm_log_stream, "xm_log: %s\n", buf);
}

struct timer {
	char label[128];
	time_t start;
};

static struct timer
timer_start(const char *fmt, ...)
{
	struct timer timer;
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(timer.label, sizeof(timer.label), fmt, ap);
	va_end(ap);

	xm_log("%s...", timer.label);
	timer.start = time(NULL);

	return (timer);
}

static void
timer_stop(struct timer *timer)
{
	time_t total = time(NULL) - timer->start;
	xm_log("%s done in %d sec", timer->label, (int)total);
}

static void
swap_ptr(xm_scalar_t **a, xm_scalar_t **b)
{
	xm_scalar_t *t = *a;
	*a = *b;
	*b = t;
}

const char *
xm_banner(void)
{
	const char *s =
"This program uses libxm tensor contraction code.\n"
"See https://github.com/ilyak/libxm for more info.\n"
"libxm is (c) Ilya Kaliman 2014-2016.\n";
	return (s);
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
xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5)
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
xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5,
    size_t dim6)
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
xm_dim_same(size_t n, size_t dim)
{
	xm_dim_t ret;

	assert(n <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < n; ret.n++)
		ret.i[ret.n] = dim;

	return (ret);
}

xm_dim_t
xm_dim_zero(size_t n)
{
	return (xm_dim_same(n, 0));
}

static xm_dim_t
xm_dim_add(const xm_dim_t *a, const xm_dim_t *b)
{
	xm_dim_t c;
	size_t i;

	assert(a->n == b->n);

	c.n = a->n;
	for (i = 0; i < c.n; i++)
		c.i[i] = a->i[i] + b->i[i];

	return (c);
}

static xm_dim_t
xm_dim_mul(const xm_dim_t *a, const xm_dim_t *b)
{
	xm_dim_t c;
	size_t i;

	assert(a->n == b->n);

	c.n = a->n;
	for (i = 0; i < c.n; i++)
		c.i[i] = a->i[i] * b->i[i];

	return (c);
}

static xm_dim_t
xm_dim_div(const xm_dim_t *a, const xm_dim_t *b)
{
	xm_dim_t c;
	size_t i;

	assert(a->n == b->n);

	c.n = a->n;
	for (i = 0; i < c.n; i++) {
		assert(a->i[i] % b->i[i] == 0);
		c.i[i] = a->i[i] / b->i[i];
	}

	return (c);
}

static int
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

static void
xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *b, const xm_dim_t *mask)
{
	size_t i;

	assert(a->n == b->n);

	for (i = 0; i < mask->n; i++)
		a->i[mask->i[i]] = b->i[mask->i[i]];
}

static void
xm_dim_zero_mask(xm_dim_t *a, const xm_dim_t *mask)
{
	size_t i;

	for (i = 0; i < mask->n; i++)
		a->i[mask->i[i]] = 0;
}

size_t
xm_dim_dot(const xm_dim_t *dim)
{
	size_t i, ret;

	ret = 1;
	for (i = 0; i < dim->n; i++)
		ret *= dim->i[i];

	return (ret);
}

static size_t
xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, ret;

	ret = 1;
	for (i = 0; i < mask->n; i++)
		ret *= dim->i[mask->i[i]];

	return (ret);
}

xm_dim_t
xm_dim_identity_permutation(size_t n)
{
	xm_dim_t ret;

	assert(n <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < n; ret.n++)
		ret.i[ret.n] = ret.n;

	return (ret);
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

static size_t
xm_dim_offset(const xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t ret = 0;

	assert(xm_dim_less(idx, dim));

	switch (idx->n) {
	case 6: ret += idx->i[5] * dim->i[4] * dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 5: ret += idx->i[4] * dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 4: ret += idx->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 3: ret += idx->i[2] * dim->i[1] * dim->i[0];
	case 2: ret += idx->i[1] * dim->i[0];
	case 1: ret += idx->i[0];
	}
	return (ret);
}

static size_t
xm_dim_offset_pdim(const xm_dim_t *aidx, const xm_dim_t *adim,
    const xm_dim_t *pidx, const xm_dim_t *pdim)
{
	xm_dim_t idx, dim;

	dim = xm_dim_mul(adim, pdim);
	idx = xm_dim_mul(pidx, adim);
	idx = xm_dim_add(&idx, aidx);

	return (xm_dim_offset(&idx, &dim));
}

static size_t
xm_dim_offset_mask(const xm_dim_t *idx, const xm_dim_t *dim,
    const xm_dim_t *mask)
{
	size_t i, offset, power;

	assert(xm_dim_less(idx, dim));

	offset = 0;
	power = 1;
	for (i = 0; i < mask->n; i++) {
		offset += idx->i[mask->i[i]] * power;
		power *= dim->i[mask->i[i]];
	}
	return (offset);
}

size_t
xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < idx->n; i++) {
		idx->i[i] += carry;
		carry = idx->i[i] / dim->i[i];
		idx->i[i] %= dim->i[i];
	}

	return (carry);
}

static size_t
xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < mask->n; i++) {
		idx->i[mask->i[i]] += carry;
		carry = idx->i[mask->i[i]] / dim->i[mask->i[i]];
		idx->i[mask->i[i]] %= dim->i[mask->i[i]];
	}

	return (carry);
}

static xm_dim_t
xm_dim_permute(const xm_dim_t *idx, const xm_dim_t *permutation)
{
	xm_dim_t ret;

	assert(idx->n == permutation->n);

	ret.n = idx->n;
	switch (ret.n) {
	case 6: ret.i[permutation->i[5]] = idx->i[5];
	case 5: ret.i[permutation->i[4]] = idx->i[4];
	case 4: ret.i[permutation->i[3]] = idx->i[3];
	case 3: ret.i[permutation->i[2]] = idx->i[2];
	case 2: ret.i[permutation->i[1]] = idx->i[1];
	case 1: ret.i[permutation->i[0]] = idx->i[0];
	}
	return (ret);
}

static xm_dim_t
xm_dim_permute_rev(const xm_dim_t *idx, const xm_dim_t *permutation)
{
	xm_dim_t ret;

	assert(idx->n == permutation->n);

	ret.n = idx->n;
	switch (ret.n) {
	case 6: ret.i[5] = idx->i[permutation->i[5]];
	case 5: ret.i[4] = idx->i[permutation->i[4]];
	case 4: ret.i[3] = idx->i[permutation->i[3]];
	case 3: ret.i[2] = idx->i[permutation->i[2]];
	case 2: ret.i[1] = idx->i[permutation->i[1]];
	case 1: ret.i[0] = idx->i[permutation->i[0]];
	}
	return (ret);
}

struct xm_tensor *
xm_tensor_create(struct xm_allocator *allocator, const xm_dim_t *dim,
    const char *label)
{
	struct xm_tensor *tensor;
	size_t i, size;

	assert(dim->n >= 1 && dim->n <= XM_MAX_DIM);

	if ((tensor = calloc(1, sizeof(*tensor))) == NULL) {
		xm_log_line("out of memory");
		return (NULL);
	}

	size = xm_dim_dot(dim);

	if ((tensor->blocks = calloc(size, sizeof(*tensor->blocks))) == NULL) {
		xm_log_line("out of memory");
		free(tensor);
		return (NULL);
	}

	for (i = 0; i < size; i++)
		tensor->blocks[i].data_ptr = XM_NULL_PTR;

	tensor->pidx = xm_dim_zero(dim->n);
	tensor->pdim = xm_dim_same(dim->n, 1);
	tensor->pdim0 = xm_dim_same(dim->n, 1);
	tensor->label = strdup(label ? label : "");
	tensor->dim = *dim;
	tensor->allocator = allocator;

	return (tensor);
}

const char *
xm_tensor_get_label(const struct xm_tensor *tensor)
{
	assert(tensor);

	return (tensor->label);
}

static struct xm_block *
xm_tensor_get_block(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	size_t offset;

	assert(tensor);
	assert(idx);

	offset = xm_dim_offset_pdim(idx, &tensor->dim, &tensor->pidx,
	    &tensor->pdim);
	return (tensor->blocks + offset);
}

void
xm_tensor_copy_data(struct xm_tensor *dst, const struct xm_tensor *src)
{
	struct xm_block *dstblk, *srcblk;
	size_t i, nblk, size;

	assert(xm_tensor_is_initialized(dst));
	assert(xm_tensor_is_initialized(src));
	assert(xm_dim_eq(&dst->dim, &src->dim));
	assert(xm_dim_eq(&dst->pdim, &src->pdim));

	nblk = xm_dim_dot(&dst->dim);

	for (i = 0; i < nblk; i++) {
		dstblk = &dst->blocks[i];
		srcblk = &src->blocks[i];
		assert(srcblk->is_source == dstblk->is_source);
		assert(srcblk->is_nonzero == dstblk->is_nonzero);
		assert(xm_dim_eq(&srcblk->dim, &dstblk->dim));
		if (dstblk->is_source) {
			size = xm_dim_dot(&dstblk->dim) * sizeof(xm_scalar_t);
			assert(srcblk->data_ptr != XM_NULL_PTR);
			xm_allocator_read(src->allocator, srcblk->data_ptr,
			    dst->block_buf, size);
			assert(dstblk->data_ptr != XM_NULL_PTR);
			xm_allocator_write(dst->allocator, dstblk->data_ptr,
			    dst->block_buf, size);
		}
	}
}

xm_scalar_t
xm_tensor_get_element(struct xm_tensor *tensor, const xm_dim_t *blk_i,
    const xm_dim_t *el_i)
{
	struct xm_block *block;
	xm_dim_t idx, dim_p;
	size_t el_off, size_bytes;

	block = xm_tensor_get_block(tensor, blk_i);
	if (!block->is_nonzero)
		return (0.0);
	size_bytes = xm_dim_dot(&block->dim) * sizeof(xm_scalar_t);
	assert(block->data_ptr != XM_NULL_PTR);
	xm_allocator_read(tensor->allocator, block->data_ptr,
	    tensor->block_buf, size_bytes);
	idx = xm_dim_permute(el_i, &block->permutation);
	dim_p = xm_dim_permute(&block->dim, &block->permutation);
	el_off = xm_dim_offset(&idx, &dim_p);
	return (block->scalar * tensor->block_buf[el_off]);
}

static void
xm_tensor_get_idx(struct xm_tensor *tensor, const xm_dim_t *abs_idx,
    xm_dim_t *blk_idx, xm_dim_t *el_idx)
{
	struct xm_block *block;
	xm_dim_t abs_dim, idx;
	size_t dim_i, blk_i, next;

	*blk_idx = xm_dim_zero(tensor->dim.n);
	*el_idx = xm_dim_zero(tensor->dim.n);

	abs_dim = xm_dim_zero(tensor->dim.n);
	for (dim_i = 0; dim_i < tensor->dim.n; dim_i++) {
		idx = xm_dim_zero(tensor->dim.n);
		for (blk_i = 0; blk_i < tensor->dim.i[dim_i]; blk_i++) {
			idx.i[dim_i] = blk_i;
			block = xm_tensor_get_block(tensor, &idx);
			next = abs_dim.i[dim_i] + block->dim.i[dim_i];
			if (next > abs_idx->i[dim_i]) {
				el_idx->i[dim_i] = abs_idx->i[dim_i] -
				    abs_dim.i[dim_i];
				break;
			}
			abs_dim.i[dim_i] = next;
			blk_idx->i[dim_i]++;
		}
		if (blk_i == tensor->dim.i[dim_i])
			assert(0);
	}
}

xm_scalar_t
xm_tensor_get_abs_element(struct xm_tensor *tensor, const xm_dim_t *idx)
{
	xm_dim_t blk_i, el_i;

	xm_tensor_get_idx(tensor, idx, &blk_i, &el_i);
	return (xm_tensor_get_element(tensor, &blk_i, &el_i));
}

int
xm_tensor_block_is_nonzero(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->is_nonzero);
}

int
xm_tensor_block_is_initialized(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->is_initialized);
}

xm_dim_t
xm_tensor_get_block_dim(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->dim);
}

uintptr_t
xm_tensor_get_block_data_ptr(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->data_ptr);
}

xm_dim_t
xm_tensor_get_block_permutation(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->permutation);
}

xm_scalar_t
xm_tensor_get_block_scalar(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->scalar);
}

static void
xm_tensor_set_block_buf(struct xm_tensor *a, const xm_dim_t *blkdim)
{
	size_t size = xm_dim_dot(blkdim) * sizeof(xm_scalar_t);

	if (size > a->block_buf_bytes) {
		if ((a->block_buf = realloc(a->block_buf, size)) == NULL) {
			xm_log_line("out of memory");
			abort();
		}
		a->block_buf_bytes = size;
	}
}

void
xm_tensor_set_zero_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim)
{
	struct xm_block *block;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(blkdim != NULL);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	block->idx = *idx;
	block->source_idx = *idx;
	block->dim = *blkdim;
	block->data_ptr = XM_NULL_PTR;
	block->permutation = xm_dim_identity_permutation(blkdim->n);
	block->scalar = 1.0;
	block->is_source = 0;
	block->is_nonzero = 0;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, blkdim);
}

void
xm_tensor_set_source_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim, uintptr_t data_ptr)
{
	struct xm_block *block;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(blkdim != NULL);
	assert(data_ptr != XM_NULL_PTR);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	block->idx = *idx;
	block->source_idx = *idx;
	block->dim = *blkdim;
	block->data_ptr = data_ptr;
	block->permutation = xm_dim_identity_permutation(blkdim->n);
	block->scalar = 1.0;
	block->is_source = 1;
	block->is_nonzero = 1;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, blkdim);
}

void
xm_tensor_set_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *source_idx, const xm_dim_t *permutation,
    xm_scalar_t scalar)
{
	struct xm_block *block, *source_block;
	xm_dim_t blkdim;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(source_idx != NULL);
	assert(permutation != NULL);

	source_block = xm_tensor_get_block(tensor, source_idx);
	assert(source_block->is_initialized);
	assert(source_block->is_source);
	assert(source_block->is_nonzero);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	blkdim = xm_dim_permute_rev(&source_block->dim, permutation);
	block->idx = *idx;
	block->source_idx = *source_idx;
	block->dim = blkdim;
	block->data_ptr = source_block->data_ptr;
	block->permutation = *permutation;
	block->scalar = scalar;
	block->is_source = 0;
	block->is_nonzero = 1;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, &blkdim);
}

int
xm_tensor_is_initialized(const struct xm_tensor *tensor)
{
	size_t i, dim;

	assert(tensor);

	dim = xm_dim_dot(&tensor->dim);

	for (i = 0; i < dim; i++)
		if (!tensor->blocks[i].is_initialized)
			return (0);
	return (1);
}

xm_dim_t
xm_tensor_get_dim(const struct xm_tensor *tensor)
{
	assert(tensor);

	return (tensor->dim);
}

void
xm_tensor_set_part_dim(struct xm_tensor *tensor, const xm_dim_t *pdim)
{
	assert(tensor);

	tensor->pdim0 = *pdim;
}

xm_dim_t
xm_tensor_get_part_dim(const struct xm_tensor *tensor)
{
	assert(tensor);

	return (tensor->pdim0);
}

xm_dim_t
xm_tensor_get_abs_dim(const struct xm_tensor *tensor)
{
	struct xm_block *block;
	xm_dim_t abs_dim, idx;
	size_t dim_i, blk_i;

	assert(xm_tensor_is_initialized(tensor));

	abs_dim = xm_dim_zero(tensor->dim.n);
	for (dim_i = 0; dim_i < tensor->dim.n; dim_i++) {
		idx = xm_dim_zero(tensor->dim.n);
		for (blk_i = 0; blk_i < tensor->dim.i[dim_i]; blk_i++) {
			idx.i[dim_i] = blk_i;
			block = xm_tensor_get_block(tensor, &idx);
			abs_dim.i[dim_i] += block->dim.i[dim_i];
		}
	}

	return (abs_dim);
}

void
xm_tensor_free(struct xm_tensor *tensor)
{
	if (tensor) {
		free(tensor->blocks);
		free(tensor->label);
		free(tensor->block_buf);
		free(tensor);
	}
}

static int
skip_idx(xm_dim_t *idx, struct xm_tensor *a, const xm_dim_t *mask,
    const bitstr_t *skip)
{
	size_t i = xm_dim_offset_mask(idx, &a->dim, mask);

	for (; bit_test(skip, i); i++)
		if (xm_dim_inc_mask(idx, &a->dim, mask))
			return (1);

	return (0);
}

static void
block_get_matrix(struct xm_block *block, xm_dim_t mask_i, xm_dim_t mask_j,
    size_t block_size_i, size_t block_size_j, xm_scalar_t *from,
    xm_scalar_t *to, size_t stride)
{
	xm_dim_t el_dim, el_dim_p, el_i, idx, permutation;
	size_t ii, jj, offset, inc, lead_ii, kk, el_dim_lead_ii;
	xm_scalar_t scalar = block->scalar;

	assert(from);
	assert(to);

	el_dim = block->dim;
	permutation = block->permutation;
	el_dim_p = xm_dim_permute(&block->dim, &permutation);
	el_i = xm_dim_zero(el_dim.n);

	inc = 1;
	el_dim_lead_ii = 1;

	if (mask_i.n > 0) {
		lead_ii = mask_i.i[0];
		for (kk = 0; kk < permutation.i[lead_ii]; kk++)
			inc *= el_dim_p.i[kk];
		for (ii = 0; ii < mask_i.n-1; ii++)
			mask_i.i[ii] = mask_i.i[ii+1];
		mask_i.n--;
		el_dim_lead_ii = el_dim.i[lead_ii];
	}

	if (inc == 1) {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&el_i, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += el_dim_lead_ii) {
				idx = xm_dim_permute(&el_i, &permutation);
				offset = xm_dim_offset(&idx, &el_dim_p);
				memcpy(&to[jj * stride + ii],
				    from + offset,
				    sizeof(xm_scalar_t) * el_dim_lead_ii);
				xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
			}
			xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
		}
	} else {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&el_i, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += el_dim_lead_ii) {
				idx = xm_dim_permute(&el_i, &permutation);
				offset = xm_dim_offset(&idx, &el_dim_p);
				for (kk = 0; kk < el_dim_lead_ii; kk++) {
					to[jj * stride + ii + kk] =
					    from[offset];
					offset += inc;
				}
				xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
			}
			xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
		}
	}

	for (jj = 0; jj < block_size_j; jj++)
		for (ii = 0; ii < block_size_i; ii++)
			to[jj * stride + ii] *= scalar;
}

static void
tensor_get_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    xm_scalar_t *data, bitstr_t *skip_i, bitstr_t *skip_j, size_t stride,
    xm_scalar_t *from)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, block_size_i, block_size_j = 0;
	xm_scalar_t *ptr, *buf = from;

	if (from == NULL) {
		/* accessed from multiple threads */
		if ((buf = malloc(tensor->block_buf_bytes)) == NULL) {
			xm_log_line("out of memory");
			abort();
		}
	}

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		ptr = data;
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			block_size_i = xm_dim_dot_mask(&block->dim, &mask_i);
			block_size_j = xm_dim_dot_mask(&block->dim, &mask_j);

			if (block->is_nonzero) {
				if (from == NULL) {
					assert(block->data_ptr != XM_NULL_PTR);
					xm_allocator_read(tensor->allocator,
					    block->data_ptr, buf,
					    block_size_i * block_size_j *
					    sizeof(xm_scalar_t));
				}
			} else {
				memset(buf, 0, block_size_i *
				    block_size_j * sizeof(xm_scalar_t));
			}
			block_get_matrix(block, mask_i, mask_j,
			    block_size_i, block_size_j,
			    buf, ptr, stride);
			if (from)
				buf += block_size_i * block_size_j;
			ptr += block_size_i;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}
		data += block_size_j * stride;

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
	if (from == NULL)
		free(buf);
}

static void *
async_tensor_get_submatrix_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_get_submatrix(async->tensor, async->blk_idx, async->mask_i,
	    async->mask_j, async->nblk_i, async->nblk_j, async->data,
	    async->skip_i, async->skip_j, async->stride,
	    async->from);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_get_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j, size_t stride,
    xm_scalar_t *from, xm_scalar_t *to)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->data = to;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->stride = stride;
	async->from = from;

	if (pthread_create(&thread, NULL,
	    async_tensor_get_submatrix_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}

static void
block_set_matrix(struct xm_block *block, xm_dim_t mask_i, xm_dim_t mask_j,
    size_t block_size_i, size_t block_size_j, xm_scalar_t *data,
    size_t stride, xm_scalar_t *buf, struct xm_allocator *allocator)
{
	xm_dim_t el_dim, el_i;
	size_t ii, jj, offset;

	if (!block->is_source)
		return;

	el_dim = block->dim;
	el_i = xm_dim_zero(el_dim.n);

	assert(mask_i.n > 0);
	assert(mask_i.i[0] == 0);
	for (ii = 0; ii < mask_i.n-1; ii++)
		mask_i.i[ii] = mask_i.i[ii+1];
	mask_i.n--;

	for (jj = 0; jj < block_size_j; jj++) {
		xm_dim_zero_mask(&el_i, &mask_i);
		for (ii = 0; ii < block_size_i; ii += el_dim.i[0]) {
			offset = xm_dim_offset(&el_i, &el_dim);
			memcpy(buf + offset,
			    &data[jj * stride + ii],
			    sizeof(xm_scalar_t) * el_dim.i[0]);
			xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
		}
		xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
	}

	assert(block->data_ptr != XM_NULL_PTR);
	xm_allocator_write(allocator, block->data_ptr, buf,
	    block_size_i * block_size_j * sizeof(xm_scalar_t));
}

static void
tensor_set_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j, xm_scalar_t *data, size_t stride)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, block_size_i, block_size_j = 0;
	xm_scalar_t *ptr, *buf;

	/* accessed from multiple threads */
	if ((buf = malloc(tensor->block_buf_bytes)) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		ptr = data;
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			block_size_i = xm_dim_dot_mask(&block->dim, &mask_i);
			block_size_j = xm_dim_dot_mask(&block->dim, &mask_j);

			block_set_matrix(block, mask_i, mask_j,
			    block_size_i, block_size_j,
			    ptr, stride, buf, tensor->allocator);
			ptr += block_size_i;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}
		data += block_size_j * stride;

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
	free(buf);
}

static void *
async_tensor_set_submatrix_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_set_submatrix(async->tensor, async->blk_idx, async->mask_i,
	    async->mask_j, async->nblk_i, async->nblk_j,
	    async->skip_i, async->skip_j, async->data, async->stride);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_set_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j,
    size_t stride, xm_scalar_t *data)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->data = data;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->stride = stride;

	if (pthread_create(&thread, NULL,
	    async_tensor_set_submatrix_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}

static int
add_chunk_blocks(struct xm_tensor *tensor, xm_dim_t *blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t cs;

	cs = 0;
	for (;;) {
		if (skip_idx(blk_idx, tensor, &mask, skip))
			return (1);
		block = xm_tensor_get_block(tensor, blk_idx);
		cs += xm_dim_dot_mask(&block->dim, &mask);
		if (cs > max_cs)
			return (0);
		if (xm_dim_inc_mask(blk_idx, &tensor->dim, &mask))
			return (1);
	}
}

static size_t
get_chunk_blocks(struct xm_tensor *tensor, xm_dim_t blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t i, cs;

	for (i = 0, cs = 0; ; i++) {
		if (skip_idx(&blk_idx, tensor, &mask, skip))
			break;
		block = xm_tensor_get_block(tensor, &blk_idx);
		cs += xm_dim_dot_mask(&block->dim, &mask);
		if (cs > max_cs)
			break;
		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask))
			return (i + 1);
	}
	return (i);
}

static size_t
get_min_chunk_size(struct xm_tensor *a, xm_dim_t mask, const bitstr_t *skip)
{
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t size, max_size;

	blk_idx = xm_dim_zero(a->dim.n);
	max_size = 0;

	for (;;) {
		if (skip_idx(&blk_idx, a, &mask, skip))
			break;
		block = xm_tensor_get_block(a, &blk_idx);
		size = xm_dim_dot_mask(&block->dim, &mask);
		if (size > max_size)
			max_size = size;
		if (xm_dim_inc_mask(&blk_idx, &a->dim, &mask))
			break;
	}
	return (max_size);
}

static size_t
get_chunk_size(struct xm_tensor *tensor, xm_dim_t blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t cs, size;

	cs = 0;
	for (;;) {
		if (skip_idx(&blk_idx, tensor, &mask, skip))
			break;
		block = xm_tensor_get_block(tensor, &blk_idx);
		size = xm_dim_dot_mask(&block->dim, &mask);
		if (cs + size > max_cs)
			break;
		cs += size;
		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask))
			break;
	}
	return (cs);
}

static bitstr_t *
make_skip(struct xm_tensor *c, xm_dim_t cidxc, xm_dim_t aidxc)
{
	bitstr_t *skip;
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t i, j, nblk_i, nblk_j;

	nblk_i = xm_dim_dot_mask(&c->dim, &cidxc);
	nblk_j = xm_dim_dot_mask(&c->dim, &aidxc);
	skip = bit_alloc(nblk_i);
	bit_nset(skip, 0, nblk_i - 1);

	blk_idx = xm_dim_zero(c->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&blk_idx, &aidxc);
		for (j = 0; j < nblk_j; j++) {
			block = xm_tensor_get_block(c, &blk_idx);
			if (block->is_source) {
				bit_clear(skip, i);
				break;
			}
			xm_dim_inc_mask(&blk_idx, &c->dim, &aidxc);
		}
		xm_dim_inc_mask(&blk_idx, &c->dim, &cidxc);
	}

	return (skip);
}

static void
set_skip_zero(bitstr_t *skip, struct xm_tensor *a, xm_dim_t cidxa,
    xm_dim_t aidxa)
{
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t i, j, nblk_i, nblk_j;
	int all_zero;

	nblk_i = xm_dim_dot_mask(&a->dim, &cidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &aidxa);

	blk_idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&blk_idx, &aidxa);
		all_zero = 1;
		for (j = 0; j < nblk_j; j++) {
			block = xm_tensor_get_block(a, &blk_idx);
			if (block->is_nonzero) {
				all_zero = 0;
				break;
			}
			xm_dim_inc_mask(&blk_idx, &a->dim, &aidxa);
		}
		if (all_zero)
			bit_set(skip, i);
		xm_dim_inc_mask(&blk_idx, &a->dim, &cidxa);
	}
}

static size_t
count_zero_bits(const bitstr_t *bits, size_t len)
{
	size_t i, cnt;

	for (i = 0, cnt = 0; i < len; i++)
		if (!bit_test(bits, i))
			cnt++;

	return (cnt);
}

static void
calc_max_chunk_size(size_t m, size_t n, size_t k, size_t *cs_m, size_t *cs_n)
{
	size_t size, cs_m2, cs_n2;

	size = get_buffer_size() / sizeof(xm_scalar_t);

	/* both a and b fit in buffer */
	cs_m2 = m;
	cs_n2 = n;
	if (k * (cs_m2 + cs_n2) + cs_m2 * cs_n2 <= size)
		goto done;

	/* a or b fits in buffer */
	if (m < n) {
		cs_m2 = m;
		if (size > cs_m2 * k) {
			cs_n2 = (size - cs_m2 * k) / (4 * k + 3 * cs_m2);
			if (cs_n2 >= cs_m2 / 4 && cs_n2 >= *cs_n)
				goto done;
		}
	} else {
		cs_n2 = n;
		if (size > cs_n2 * k) {
			cs_m2 = (size - cs_n2 * k) / (4 * k + 3 * cs_n2);
			if (cs_m2 >= cs_n2 / 4 && cs_m2 >= *cs_m)
				goto done;
		}
	}

	/* neither a nor b fits in buffer */
	cs_m2 = *cs_m;
	cs_n2 = *cs_n;
	while (4 * k * (cs_m2 + cs_n2) + 3 * cs_m2 * cs_n2 < size) {
		cs_m2++;
		cs_n2++;
	}
done:
	assert(cs_m2 <= m);
	assert(cs_n2 <= n);
	assert(cs_m2 >= *cs_m);
	assert(cs_n2 >= *cs_n);
	*cs_m = cs_m2;
	*cs_n = cs_n2;
}

static void
tensor_prefetch(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j, xm_scalar_t *data)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, size_bytes, size;

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			size = xm_dim_dot(&block->dim);
			if (block->is_nonzero) {
				size_bytes = size * sizeof(xm_scalar_t);
				assert(block->data_ptr != XM_NULL_PTR);
				xm_allocator_read(tensor->allocator,
				    block->data_ptr, data, size_bytes);
			}
			data += size;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
}

static void *
async_tensor_prefetch_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_prefetch(async->tensor, async->blk_idx,
	    async->mask_i, async->mask_j, async->nblk_i, async->nblk_j,
	    async->skip_i, async->skip_j, async->data);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_prefetch(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j, xm_scalar_t *data)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->data = data;

	if (pthread_create(&thread, NULL,
	    async_tensor_prefetch_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}

static pthread_t
async_tensor_prefetch_next(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    bitstr_t *skip_i, bitstr_t *skip_j, xm_scalar_t *data, size_t max_cs_m,
    int wrap)
{
	int done;

	done = add_chunk_blocks(tensor, &blk_idx, mask_j,
	    skip_j, max_cs_m);
	if (wrap || !done) {
		nblk_j = get_chunk_blocks(tensor, blk_idx, mask_j,
		    skip_j, max_cs_m);
		return (async_tensor_prefetch(tensor, blk_idx, mask_i, mask_j,
		    nblk_i, nblk_j, skip_i, skip_j, data));
	}
	return (pthread_self());
}

static void
check_block_consistency(struct xm_tensor *t1, struct xm_tensor *t2,
    xm_dim_t mask1, xm_dim_t mask2)
{
	struct xm_block *blk1, *blk2;
	xm_dim_t idx1, idx2;
	size_t i, n, dim1, dim2;

	n = xm_dim_dot_mask(&t1->dim, &mask1);
	assert(n == xm_dim_dot_mask(&t2->dim, &mask2));

	idx1 = xm_dim_zero(t1->dim.n);
	idx2 = xm_dim_zero(t2->dim.n);
	for (i = 0; i < n; i++) {
		blk1 = xm_tensor_get_block(t1, &idx1);
		blk2 = xm_tensor_get_block(t2, &idx2);
		dim1 = xm_dim_dot_mask(&blk1->dim, &mask1);
		dim2 = xm_dim_dot_mask(&blk2->dim, &mask2);
		assert(dim1 == dim2);
		xm_dim_inc_mask(&idx1, &t1->dim, &mask1);
		xm_dim_inc_mask(&idx2, &t2->dim, &mask2);
	}
}

static void
parse_idx(const char *str1, const char *str2, xm_dim_t *mask1, xm_dim_t *mask2)
{
	size_t len1, len2, i, j;

	len1 = strlen(str1);
	len2 = strlen(str2);

	mask1->n = 0;
	mask2->n = 0;
	for (i = 0; i < len1; i++) {
		for (j = 0; j < len2; j++) {
			if (str1[i] == str2[j]) {
				mask1->i[mask1->n++] = i;
				mask2->i[mask2->n++] = j;
			}
		}
	}
}

static int
has_k_symmetry(struct xm_tensor *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2)
{
	struct xm_block *blk, *blk2;
	xm_dim_t idx, idx2;
	size_t i, j, nblk_i, nblk_j;

	if (a->dim.i[cidxa.i[si1]] != a->dim.i[cidxa.i[si2]])
		return (0);

	nblk_i = xm_dim_dot_mask(&a->dim, &cidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &aidxa);

	idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &aidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
			if (idx.i[cidxa.i[si1]] != idx.i[cidxa.i[si2]]) {
				idx2 = idx;
				idx2.i[cidxa.i[si1]] = idx.i[cidxa.i[si2]];
				idx2.i[cidxa.i[si2]] = idx.i[cidxa.i[si1]];
				blk2 = xm_tensor_get_block(a, &idx2);
				if (blk->is_nonzero || blk2->is_nonzero) {
					if (!xm_dim_eq(&blk->source_idx,
						       &blk2->source_idx))
						return (0);
				}
			}
			xm_dim_inc_mask(&idx, &a->dim, &aidxa);
		}
		xm_dim_inc_mask(&idx, &a->dim, &cidxa);
	}

	return (1);
}

static void
set_k_symmetry(struct xm_tensor *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2, int enable)
{
	struct xm_block *blk;
	xm_dim_t idx;
	size_t nblk_i, nblk_j, i, j;

	nblk_i = xm_dim_dot_mask(&a->dim, &aidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &cidxa);

	idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &cidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
			if (idx.i[cidxa.i[si1]] < idx.i[cidxa.i[si2]])
				blk->scalar *= enable ? 2.0 : 0.5;
			else if (idx.i[cidxa.i[si1]] > idx.i[cidxa.i[si2]]) {
				blk->is_nonzero = !enable &&
				    blk->data_ptr != XM_NULL_PTR;
			}
			xm_dim_inc_mask(&idx, &a->dim, &cidxa);
		}
		xm_dim_inc_mask(&idx, &a->dim, &aidxa);
	}
}

static int
xm_contract_part(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    xm_scalar_t beta, struct xm_tensor *c, xm_dim_t cidxa, xm_dim_t aidxa,
    xm_dim_t cidxb, xm_dim_t aidxb, xm_dim_t cidxc, xm_dim_t aidxc)
{
	bitstr_t *skip_m, *skip_n, *skip_k;
	xm_scalar_t *blk_a1, *blk_a2, *blk_a3, *blk_a4;
	xm_scalar_t *blk_b1, *blk_b2, *blk_b3, *blk_b4;
	xm_scalar_t *buf, *blk_c1, *blk_c2, *blk_c3;
	int done_a, done_b, get_c, split_a, split_b;
	pthread_t thr_a1, thr_a2, thr_b1, thr_b2, thr_c1, thr_c2;
	size_t m, n, k, max_cs_m, max_cs_n;
	size_t nblk_k, blk_cs_m, blk_cs_n;
	size_t size, size_a, size_b, size_c;
	struct timer gemm_timer;
	xm_dim_t blk_ia, blk_ib, blk_ic, blk_ic2;

	get_c = beta != 0.0;

	blk_ia = xm_dim_zero(a->dim.n);
	blk_ib = xm_dim_zero(b->dim.n);
	blk_ic = xm_dim_zero(c->dim.n);

	nblk_k = xm_dim_dot_mask(&a->dim, &cidxa);

	skip_m = make_skip(c, cidxc, aidxc);
	skip_n = make_skip(c, aidxc, cidxc);
	skip_k = bit_alloc(nblk_k);
	bit_nclear(skip_k, 0, nblk_k - 1);
	set_skip_zero(skip_k, a, cidxa, aidxa);
	set_skip_zero(skip_k, b, cidxb, aidxb);

	nblk_k = count_zero_bits(skip_k, nblk_k);

	m = get_chunk_size(c, blk_ic, cidxc, skip_m, ULONG_MAX);
	n = get_chunk_size(c, blk_ic, aidxc, skip_n, ULONG_MAX);
	k = get_chunk_size(a, blk_ia, cidxa, skip_k, ULONG_MAX);

	if (m == 0 || n == 0 || k == 0) {
		free(skip_m);
		free(skip_n);
		free(skip_k);
		return (XM_RESULT_SUCCESS);
	}

	max_cs_m = get_min_chunk_size(c, cidxc, skip_m);
	max_cs_n = get_min_chunk_size(c, aidxc, skip_n);
	calc_max_chunk_size(m, n, k, &max_cs_m, &max_cs_n);

	split_a = max_cs_m < m;
	split_b = max_cs_n < n;
	size_a = k * max_cs_m;
	size_b = k * max_cs_n;
	size_c = max_cs_m * max_cs_n;
	size = size_a + size_b + size_c;
	if (split_a)
		size += 3 * size_a;
	if (split_b)
		size += 3 * size_b;
	if (split_a || split_b)
		size += 2 * size_c;

	xm_log("m=%zu n=%zu k=%zu max_cs_m=%zu max_cs_n=%zu",
	    m, n, k, max_cs_m, max_cs_n);
	xm_log("allocating %zu mb buffer",
	    size * sizeof(xm_scalar_t) / 1024 / 1024);

	if ((buf = malloc(size * sizeof(xm_scalar_t))) == NULL) {
		xm_log_line("out of memory");
		free(skip_m);
		free(skip_n);
		free(skip_k);
		return (XM_RESULT_NO_MEMORY);
	}

	blk_a1 = buf;
	blk_a2 = split_a ? blk_a1 + size_a : blk_a1;
	blk_a3 = split_a ? blk_a2 + size_a : blk_a2;
	blk_a4 = split_a ? blk_a3 + size_a : blk_a2;
	blk_b1 = blk_a4 + size_a;
	blk_b2 = split_b ? blk_b1 + size_b : blk_b1;
	blk_b3 = split_b ? blk_b2 + size_b : blk_b2;
	blk_b4 = split_b ? blk_b3 + size_b : blk_b2;
	blk_c1 = blk_b4 + size_b;
	blk_c2 = split_a || split_b ? blk_c1 + size_c : blk_c1;
	blk_c3 = split_a || split_b ? blk_c2 + size_c : blk_c1;

	blk_cs_m = get_chunk_blocks(c, blk_ic, cidxc, skip_m, max_cs_m);
	blk_cs_n = get_chunk_blocks(c, blk_ic, aidxc, skip_n, max_cs_n);

	thr_a1 = async_tensor_get_submatrix(a, blk_ia, cidxa, aidxa,
	    nblk_k, blk_cs_m, skip_k, skip_m, k, NULL, blk_a2);
	thr_a2 = async_tensor_prefetch_next(a, blk_ia, cidxa, aidxa,
	    nblk_k, blk_cs_m, skip_k, skip_m, blk_a4, max_cs_m, 0);
	thr_b1 = async_tensor_get_submatrix(b, blk_ib, cidxb, aidxb,
	    nblk_k, blk_cs_n, skip_k, skip_n, k, NULL, blk_b2);
	thr_b2 = async_tensor_prefetch_next(b, blk_ib, cidxb, aidxb,
	    nblk_k, blk_cs_n, skip_k, skip_n, blk_b4, max_cs_n, split_b);
	thr_c1 = pthread_self();
	thr_c2 = pthread_self();

	if (get_c) {
		if (aidxc.n > 0 && aidxc.i[0] == 0) {
			thr_c1 = async_tensor_get_submatrix(c, blk_ic,
			    aidxc, cidxc, blk_cs_n, blk_cs_m,
			    skip_n, skip_m, max_cs_n, NULL, blk_c2);
		} else {
			thr_c1 = async_tensor_get_submatrix(c, blk_ic,
			    cidxc, aidxc, blk_cs_m, blk_cs_n,
			    skip_m, skip_n, max_cs_m, NULL, blk_c2);
		}
	}

	done_a = 0;
	while (!done_a) {
		if (pthread_join(thr_a1, NULL))
			xm_log_perror("pthread_join");
		swap_ptr(&blk_a1, &blk_a2);

		if (!pthread_equal(thr_a2, pthread_self()))
			if (pthread_join(thr_a2, NULL))
				xm_log_perror("pthread_join");
		swap_ptr(&blk_a3, &blk_a4);

		done_a = add_chunk_blocks(a, &blk_ia, aidxa,
		    skip_m, max_cs_m);
		if (!done_a) {
			blk_cs_m = get_chunk_blocks(a, blk_ia, aidxa,
			    skip_m, max_cs_m);
			thr_a1 = async_tensor_get_submatrix(a,
			    blk_ia, cidxa, aidxa, nblk_k, blk_cs_m,
			    skip_k, skip_m, k, blk_a3, blk_a2);
			thr_a2 = async_tensor_prefetch_next(a, blk_ia,
			    cidxa, aidxa, nblk_k, blk_cs_m,
			    skip_k, skip_m, blk_a4, max_cs_m, 0);
		}

		blk_ib = xm_dim_zero(b->dim.n);
		done_b = 0;
		while (!done_b) {
			if (!pthread_equal(thr_b1, pthread_self()))
				if (pthread_join(thr_b1, NULL))
					xm_log_perror("pthread_join");
			swap_ptr(&blk_b1, &blk_b2);

			if (!pthread_equal(thr_b2, pthread_self()))
				if (pthread_join(thr_b2, NULL))
					xm_log_perror("pthread_join");
			swap_ptr(&blk_b3, &blk_b4);

			done_b = add_chunk_blocks(b, &blk_ib, aidxb,
			    skip_n, max_cs_n);
			if (split_b && (!done_a || !done_b)) {
				blk_cs_n = get_chunk_blocks(b, blk_ib, aidxb,
				    skip_n, max_cs_n);
				thr_b1 = async_tensor_get_submatrix(b,
				    blk_ib, cidxb, aidxb, nblk_k, blk_cs_n,
				    skip_k, skip_n, k, blk_b3, blk_b2);
				thr_b2 = async_tensor_prefetch_next(b, blk_ib,
				    cidxb, aidxb, nblk_k, blk_cs_n,
				    skip_k, skip_n, blk_b4, max_cs_n,
				    split_b && !done_a);
			} else {
				thr_b1 = pthread_self();
				thr_b2 = pthread_self();
			}

			if (get_c) {
				if (pthread_join(thr_c1, NULL))
					xm_log_perror("pthread_join");
			}
			swap_ptr(&blk_c1, &blk_c2);
			blk_ic2 = blk_ic;

			if (!done_b) {
				add_chunk_blocks(c, &blk_ic, aidxc,
				    skip_n, max_cs_n);
			} else if (!done_a) {
				xm_dim_zero_mask(&blk_ic, &aidxc);
				add_chunk_blocks(c, &blk_ic, cidxc,
				    skip_m, max_cs_m);
			}

			if (get_c && (!done_a || !done_b)) {
				assert(blk_c2 != blk_c1);
				blk_cs_m = get_chunk_blocks(c, blk_ic, cidxc,
				    skip_m, max_cs_m);
				blk_cs_n = get_chunk_blocks(c, blk_ic, aidxc,
				    skip_n, max_cs_n);
				assert(blk_cs_m > 0);
				assert(blk_cs_n > 0);
				if (aidxc.n > 0 && aidxc.i[0] == 0) {
					thr_c1 = async_tensor_get_submatrix(c,
					    blk_ic, aidxc, cidxc, blk_cs_n,
					    blk_cs_m, skip_n,
					    skip_m, max_cs_n, NULL, blk_c2);
				} else {
					thr_c1 = async_tensor_get_submatrix(c,
					    blk_ic, cidxc, aidxc, blk_cs_m,
					    blk_cs_n, skip_m,
					    skip_n, max_cs_m, NULL, blk_c2);
				}
			}

			m = get_chunk_size(c, blk_ic2, cidxc, skip_m, max_cs_m);
			n = get_chunk_size(c, blk_ic2, aidxc, skip_n, max_cs_n);

			gemm_timer = timer_start("blas_gemm");
			xm_log("blas_gemm m=%zu n=%zu k=%zu", m, n, k);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				gemm_wrapper('T', 'N', (int)n, (int)m, (int)k,
				    alpha, blk_b1, (int)k, blk_a1, (int)k,
				    beta, blk_c1, (int)max_cs_n);
			} else {
				gemm_wrapper('T', 'N', (int)m, (int)n, (int)k,
				    alpha, blk_a1, (int)k, blk_b1, (int)k,
				    beta, blk_c1, (int)max_cs_m);
			}
			timer_stop(&gemm_timer);

			if (!pthread_equal(thr_c2, pthread_self())) {
				if (pthread_join(thr_c2, NULL))
					xm_log_perror("pthread_join");
			}
			swap_ptr(&blk_c1, &blk_c3);
			blk_cs_m = get_chunk_blocks(c, blk_ic2, cidxc,
			    skip_m, max_cs_m);
			blk_cs_n = get_chunk_blocks(c, blk_ic2, aidxc,
			    skip_n, max_cs_n);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				thr_c2 = async_tensor_set_submatrix(c, blk_ic2,
				    aidxc, cidxc, blk_cs_n, blk_cs_m,
				    skip_n, skip_m, max_cs_n, blk_c3);
			} else {
				thr_c2 = async_tensor_set_submatrix(c, blk_ic2,
				    cidxc, aidxc, blk_cs_m, blk_cs_n,
				    skip_m, skip_n, max_cs_m, blk_c3);
			}
		}
	}
	if (pthread_join(thr_c2, NULL))
		xm_log_perror("pthread_join");
	free(buf);
	free(skip_m);
	free(skip_n);
	free(skip_k);
	return (XM_RESULT_SUCCESS);
}

int
xm_contract(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    xm_scalar_t beta, struct xm_tensor *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	struct timer timer, timer2;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc;
	xm_dim_t dim_a, dim_b, dim_c;
	size_t i, j, si1, si2, npart_m, npart_n;
	int res, sym_k;

	assert(xm_tensor_is_initialized(a));
	assert(xm_tensor_is_initialized(b));
	assert(xm_tensor_is_initialized(c));
	assert(strlen(idxa) == a->dim.n);
	assert(strlen(idxb) == b->dim.n);
	assert(strlen(idxc) == c->dim.n);

	res = XM_RESULT_SUCCESS;

	parse_idx(idxa, idxb, &cidxa, &cidxb);
	parse_idx(idxc, idxa, &cidxc, &aidxa);
	parse_idx(idxc, idxb, &aidxc, &aidxb);

	dim_a = xm_tensor_get_abs_dim(a);
	dim_b = xm_tensor_get_abs_dim(b);
	i = xm_dim_dot_mask(&dim_a, &aidxa);
	j = xm_dim_dot_mask(&dim_b, &aidxb);
	if (i < j) { /* swap a and b */
		struct xm_tensor *t;
		parse_idx(idxb, idxa, &cidxa, &cidxb);
		parse_idx(idxc, idxb, &cidxc, &aidxa);
		parse_idx(idxc, idxa, &aidxc, &aidxb);
		t = a, a = b, b = t;
	}

	assert(aidxa.n + cidxa.n == a->dim.n);
	assert(aidxb.n + cidxb.n == b->dim.n);
	assert(aidxc.n + cidxc.n == c->dim.n);
	assert((aidxc.n > 0 && aidxc.i[0] == 0) ||
	       (cidxc.n > 0 && cidxc.i[0] == 0));

	check_block_consistency(a, b, cidxa, cidxb);
	check_block_consistency(c, a, cidxc, aidxa);
	check_block_consistency(c, b, aidxc, aidxb);

	sym_k = 0;
	si2 = 0;

	for (si1 = 0; si1 < cidxa.n; si1++) {
		for (si2 = si1+1; si2 < cidxa.n; si2++) {
			sym_k = has_k_symmetry(a, cidxa, aidxa, si1, si2) &&
				has_k_symmetry(b, cidxb, aidxb, si1, si2);
			if (sym_k)
				break;
		}
		if (sym_k)
			break;
	}

	if (sym_k) {
		xm_log("enabling k symmetry");
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 1);
	}

	dim_a = a->dim;
	dim_b = b->dim;
	dim_c = c->dim;

	a->pdim = a->pdim0;
	b->pdim = b->pdim0;
	c->pdim = c->pdim0;

	npart_m = xm_dim_dot_mask(&c->pdim, &cidxc);
	npart_n = xm_dim_dot_mask(&c->pdim, &aidxc);
	assert(xm_dim_dot_mask(&a->dim, &aidxa) % npart_m == 0);
	assert(xm_dim_dot_mask(&b->dim, &aidxb) % npart_n == 0);

	for (i = 0; i < cidxa.n; i++)
		a->pdim.i[cidxa.i[i]] = 1;
	for (i = 0; i < cidxb.n; i++)
		b->pdim.i[cidxb.i[i]] = 1;
	for (i = 0; i < aidxa.n; i++)
		a->pdim.i[aidxa.i[i]] = c->pdim.i[cidxc.i[i]];
	for (i = 0; i < aidxb.n; i++)
		b->pdim.i[aidxb.i[i]] = c->pdim.i[aidxc.i[i]];

	a->dim = xm_dim_div(&a->dim, &a->pdim);
	b->dim = xm_dim_div(&b->dim, &b->pdim);
	c->dim = xm_dim_div(&c->dim, &c->pdim);

	a->pidx = xm_dim_zero(a->pdim.n);
	b->pidx = xm_dim_zero(b->pdim.n);
	c->pidx = xm_dim_zero(c->pdim.n);

	timer = timer_start("xm_contract");
	for (i = 0; i < npart_m; i++) {
		xm_dim_zero_mask(&b->pidx, &aidxb);
		xm_dim_zero_mask(&c->pidx, &aidxc);
		for (j = 0; j < npart_n; j++) {
			timer2 = timer_start("xm_contract_part");
			if ((res = xm_contract_part(alpha, a, b, beta, c,
			    cidxa, aidxa, cidxb, aidxb, cidxc, aidxc)))
				goto error;
			timer_stop(&timer2);

			xm_dim_inc_mask(&b->pidx, &b->pdim, &aidxb);
			xm_dim_inc_mask(&c->pidx, &c->pdim, &aidxc);
		}
		xm_dim_inc_mask(&a->pidx, &a->pdim, &aidxa);
		xm_dim_inc_mask(&c->pidx, &c->pdim, &cidxc);
	}
	timer_stop(&timer);
error:
	if (sym_k) {
		xm_log("disabling k symmetry");
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 0);
	}
	a->dim = dim_a;
	b->dim = dim_b;
	c->dim = dim_c;
	a->pdim = xm_dim_same(a->dim.n, 1);
	b->pdim = xm_dim_same(b->dim.n, 1);
	c->pdim = xm_dim_same(c->dim.n, 1);
	a->pidx = xm_dim_zero(a->dim.n);
	b->pidx = xm_dim_zero(b->dim.n);
	c->pidx = xm_dim_zero(c->dim.n);

	return (res);
}
