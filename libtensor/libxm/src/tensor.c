/*
 * Copyright (c) 2014-2018 Ilya Kaliman
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "util.h"

struct xm_block {
	xm_block_type_t type;
	xm_dim_t permutation;
	xm_scalar_t scalar;
	uint64_t data_ptr; /* for derivative blocks stores offset of the
			      corresponding canonical block */
};

struct xm_tensor {
	xm_scalar_type_t type;
	xm_block_space_t *bs;
	xm_allocator_t *allocator;
	struct xm_block *blocks;
};

static struct xm_block *
tensor_get_block(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	xm_dim_t nblocks;
	size_t offset;

	nblocks = xm_tensor_get_nblocks(tensor);
	offset = xm_dim_offset(&blkidx, &nblocks);

	return (&tensor->blocks[offset]);
}

xm_tensor_t *
xm_tensor_create(const xm_block_space_t *bs, xm_scalar_type_t type,
    xm_allocator_t *allocator)
{
	xm_dim_t idx, nblocks;
	xm_tensor_t *ret;

	assert(bs);
	assert(allocator);

	if (!xm_scalar_check_type(type))
		fatal("unexpected scalar type");
	if ((ret = calloc(1, sizeof *ret)) == NULL)
		fatal("out of memory");
	if ((ret->bs = xm_block_space_clone(bs)) == NULL)
		fatal("out of memory");
	ret->type = type;
	ret->allocator = allocator;
	nblocks = xm_block_space_get_nblocks(bs);
	if ((ret->blocks = calloc(xm_dim_dot(&nblocks),
	    sizeof *ret->blocks)) == NULL)
		fatal("out of memory");
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		xm_tensor_set_zero_block(ret, idx);
		xm_dim_inc(&idx, &nblocks);
	}
	return ret;
}

xm_tensor_t *
xm_tensor_create_canonical(const xm_block_space_t *bs, xm_scalar_type_t type,
    xm_allocator_t *allocator)
{
	xm_tensor_t *ret;
	xm_dim_t idx, nblocks;

	ret = xm_tensor_create(bs, type, allocator);
	nblocks = xm_block_space_get_nblocks(bs);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		xm_tensor_set_canonical_block(ret, idx);
		xm_dim_inc(&idx, &nblocks);
	}
	return ret;
}

xm_tensor_t *
xm_tensor_create_structure(const xm_tensor_t *tensor, xm_scalar_type_t type,
    xm_allocator_t *allocator)
{
	xm_tensor_t *ret;
	xm_dim_t idx, nblocks;

	if (allocator == NULL)
		allocator = xm_tensor_get_allocator(tensor);
	ret = xm_tensor_create(tensor->bs, type, allocator);
	nblocks = xm_tensor_get_nblocks(ret);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		size_t i = xm_dim_offset(&idx, &nblocks);
		ret->blocks[i] = tensor->blocks[i];
		if (ret->blocks[i].type == XM_BLOCK_TYPE_CANONICAL) {
			ret->blocks[i].type = XM_BLOCK_TYPE_ZERO;
			xm_tensor_set_canonical_block(ret, idx);
		}
		xm_dim_inc(&idx, &nblocks);
	}
	return ret;
}

const xm_block_space_t *
xm_tensor_get_block_space(const xm_tensor_t *tensor)
{
	return tensor->bs;
}

xm_scalar_type_t
xm_tensor_get_scalar_type(const xm_tensor_t *tensor)
{
	return tensor->type;
}

xm_allocator_t *
xm_tensor_get_allocator(const xm_tensor_t *tensor)
{
	return tensor->allocator;
}

xm_dim_t
xm_tensor_get_abs_dims(const xm_tensor_t *tensor)
{
	return xm_block_space_get_abs_dims(tensor->bs);
}

xm_dim_t
xm_tensor_get_nblocks(const xm_tensor_t *tensor)
{
	return xm_block_space_get_nblocks(tensor->bs);
}

xm_scalar_t
xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx)
{
	struct xm_block *block;
	xm_dim_t blkidx, blkdims, elidx;
	size_t eloff, blkbytes;
	xm_scalar_t ret = 0;
	void *buf;

	xm_block_space_decompose_index(tensor->bs, idx, &blkidx, &elidx);
	block = tensor_get_block(tensor, blkidx);
	if (block->type == XM_BLOCK_TYPE_ZERO)
		return ret;
	elidx = xm_dim_permute(&elidx, &block->permutation);
	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	blkbytes = xm_dim_dot(&blkdims) * xm_scalar_sizeof(tensor->type);
	blkdims = xm_dim_permute(&blkdims, &block->permutation);
	eloff = xm_dim_offset(&elidx, &blkdims);
	if ((buf = malloc(blkbytes)) == NULL)
		fatal("out of memory");
	xm_tensor_read_block(tensor, blkidx, buf);
	ret = xm_scalar_get_element(buf, eloff, tensor->type);
	free(buf);
	return xm_scalar_mul(block->scalar, ret, tensor->type);
}

xm_block_type_t
xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return tensor_get_block(tensor, blkidx)->type;
}

xm_dim_t
xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return xm_block_space_get_block_dims(tensor->bs, blkidx);
}

size_t
xm_tensor_get_block_size(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return xm_block_space_get_block_size(tensor->bs, blkidx);
}

size_t
xm_tensor_get_block_bytes(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return xm_tensor_get_block_size(tensor, blkidx) *
	       xm_scalar_sizeof(tensor->type);
}

size_t
xm_tensor_get_largest_block_size(const xm_tensor_t *tensor)
{
	return xm_block_space_get_largest_block_size(tensor->bs);
}

size_t
xm_tensor_get_largest_block_bytes(const xm_tensor_t *tensor)
{
	return xm_tensor_get_largest_block_size(tensor) *
	       xm_scalar_sizeof(tensor->type);
}

uint64_t
xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = tensor_get_block(tensor, blkidx);
	if (block->type == XM_BLOCK_TYPE_CANONICAL)
		return block->data_ptr;
	if (block->type == XM_BLOCK_TYPE_DERIVATIVE)
		return tensor->blocks[block->data_ptr].data_ptr;
	return XM_NULL_PTR;
}

xm_dim_t
xm_tensor_get_block_permutation(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return tensor_get_block(tensor, blkidx)->permutation;
}

xm_scalar_t
xm_tensor_get_block_scalar(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return tensor_get_block(tensor, blkidx)->scalar;
}

void
xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_ZERO;
	block->permutation = xm_dim_identity_permutation(blkidx.n);
	block->scalar = 0;
	block->data_ptr = XM_NULL_PTR;
}

void
xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	size_t blkbytes;
	uint64_t data_ptr;

	if (xm_tensor_get_block_type(tensor, blkidx) != XM_BLOCK_TYPE_ZERO)
		fatal("block must be zero");
	blkbytes = xm_tensor_get_block_bytes(tensor, blkidx);
	data_ptr = xm_allocator_allocate(tensor->allocator, blkbytes);
	if (data_ptr == XM_NULL_PTR)
		fatal("unable to allocate block data");
	xm_tensor_set_canonical_block_raw(tensor, blkidx, data_ptr);
}

void
xm_tensor_set_canonical_block_raw(xm_tensor_t *tensor, xm_dim_t blkidx,
    uint64_t data_ptr)
{
	struct xm_block *block;

	if (data_ptr == XM_NULL_PTR)
		fatal("unexpected null data pointer");
	if (xm_tensor_get_block_type(tensor, blkidx) != XM_BLOCK_TYPE_ZERO)
		fatal("block must be zero");
	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_CANONICAL;
	block->permutation = xm_dim_identity_permutation(blkidx.n);
	block->scalar = 1;
	block->data_ptr = data_ptr;
}

void
xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar)
{
	struct xm_block *block;
	xm_dim_t blkdims1, blkdims2, nblocks;
	xm_block_type_t blocktype;

	if (xm_tensor_get_block_type(tensor, blkidx) != XM_BLOCK_TYPE_ZERO)
		fatal("block must be zero");
	blocktype = xm_tensor_get_block_type(tensor, source_blkidx);
	if (blocktype != XM_BLOCK_TYPE_CANONICAL)
		fatal("derivative blocks must have canonical source blocks");
	blkdims1 = xm_block_space_get_block_dims(tensor->bs, blkidx);
	blkdims1 = xm_dim_permute(&blkdims1, &permutation);
	blkdims2 = xm_block_space_get_block_dims(tensor->bs, source_blkidx);
	if (xm_dim_ne(&blkdims1, &blkdims2))
		fatal("invalid block permutation");
	nblocks = xm_tensor_get_nblocks(tensor);
	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_DERIVATIVE;
	block->permutation = permutation;
	block->scalar = scalar;
	block->data_ptr = xm_dim_offset(&source_blkidx, &nblocks);
}

void
xm_tensor_get_canonical_block_list(const xm_tensor_t *tensor,
    xm_dim_t **blklist, size_t *nblklist)
{
	xm_dim_t idx, nblocks, *list = NULL;
	xm_block_type_t blocktype;
	size_t nlist = 0;

	nblocks = xm_tensor_get_nblocks(tensor);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		blocktype = xm_tensor_get_block_type(tensor, idx);
		if (blocktype == XM_BLOCK_TYPE_CANONICAL) {
			nlist++;
			list = realloc(list, nlist * sizeof *list);
			if (list == NULL)
				fatal("out of memory");
			list[nlist-1] = idx;
		}
		xm_dim_inc(&idx, &nblocks);
	}
	*blklist = list;
	*nblklist = nlist;
}

void
xm_tensor_read_block(const xm_tensor_t *tensor, xm_dim_t blkidx, void *buf)
{
	size_t blkbytes;
	uint64_t data_ptr;
	xm_block_type_t blocktype;

	blocktype = xm_tensor_get_block_type(tensor, blkidx);
	if (blocktype == XM_BLOCK_TYPE_ZERO)
		fatal("cannot read data from zero-blocks");
	blkbytes = xm_tensor_get_block_bytes(tensor, blkidx);
	data_ptr = xm_tensor_get_block_data_ptr(tensor, blkidx);
	xm_allocator_read(tensor->allocator, data_ptr, buf, blkbytes);
}

void
xm_tensor_write_block(xm_tensor_t *tensor, xm_dim_t blkidx, const void *buf)
{
	size_t blkbytes;
	uint64_t data_ptr;
	xm_block_type_t blocktype;

	blocktype = xm_tensor_get_block_type(tensor, blkidx);
	if (blocktype != XM_BLOCK_TYPE_CANONICAL)
		fatal("can only write to canonical blocks");
	blkbytes = xm_tensor_get_block_bytes(tensor, blkidx);
	data_ptr = xm_tensor_get_block_data_ptr(tensor, blkidx);
	xm_allocator_write(tensor->allocator, data_ptr, buf, blkbytes);
}

typedef void (*kernel_fn_t)(void *, const void *, size_t, size_t, size_t,
    size_t, size_t, size_t);

static void
fold_kernel_memcpy(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	memcpy((char *)to + offset * size,
	    (const char *)from + (j * stride + i) * size,
	    lead_ii_nel * size);
}

static void
fold_kernel_float(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	float *xto = to;
	const float *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[offset] = xfrom[j * stride + i + k];
		offset += size;
	}
}

static void
fold_kernel_float_complex(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	float complex *xto = to;
	const float complex *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[offset] = xfrom[j * stride + i + k];
		offset += size;
	}
}

static void
fold_kernel_double(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	double *xto = to;
	const double *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[offset] = xfrom[j * stride + i + k];
		offset += size;
	}
}

static void
fold_kernel_double_complex(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	double complex *xto = to;
	const double complex *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[offset] = xfrom[j * stride + i + k];
		offset += size;
	}
}

void
xm_tensor_fold_block(const xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const void *from, void *to,
    size_t stride)
{
	kernel_fn_t kernel_fn;
	xm_dim_t blkdims, elidx;
	size_t ii, jj, kk, offset, inc, lead_ii, lead_ii_nel;
	size_t block_size_i, block_size_j, size;
	xm_block_type_t blocktype;

	if (from == NULL || to == NULL || from == to)
		fatal("invalid argument");
	if (mask_i.n + mask_j.n != blkidx.n)
		fatal("invalid mask dimensions");

	blocktype = xm_tensor_get_block_type(tensor, blkidx);
	if (blocktype != XM_BLOCK_TYPE_CANONICAL)
		fatal("can only fold canonical blocks");

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	block_size_i = xm_dim_dot_mask(&blkdims, &mask_i);
	block_size_j = xm_dim_dot_mask(&blkdims, &mask_j);
	elidx = xm_dim_zero(blkdims.n);

	inc = 1;
	lead_ii_nel = 1;

	if (mask_i.n > 0) {
		lead_ii = mask_i.i[0];
		for (kk = 0; kk < lead_ii; kk++)
			inc *= blkdims.i[kk];
		for (ii = 0; ii < mask_i.n-1; ii++)
			mask_i.i[ii] = mask_i.i[ii+1];
		mask_i.n--;
		lead_ii_nel = blkdims.i[lead_ii];
	}
	if (inc == 1) {
		size = xm_scalar_sizeof(tensor->type);
		kernel_fn = fold_kernel_memcpy;
	} else {
		size = inc;
		switch (tensor->type) {
		case XM_SCALAR_FLOAT:
			kernel_fn = fold_kernel_float;
			break;
		case XM_SCALAR_FLOAT_COMPLEX:
			kernel_fn = fold_kernel_float_complex;
			break;
		case XM_SCALAR_DOUBLE:
			kernel_fn = fold_kernel_double;
			break;
		case XM_SCALAR_DOUBLE_COMPLEX:
			kernel_fn = fold_kernel_double_complex;
			break;
		default:
			fatal("unexpected scalar type");
		}
	}
	for (jj = 0; jj < block_size_j; jj++) {
		xm_dim_zero_mask(&elidx, &mask_i);
		for (ii = 0; ii < block_size_i; ii += lead_ii_nel) {
			offset = xm_dim_offset(&elidx, &blkdims);
			kernel_fn(to, from, ii, jj, offset, stride, size,
			    lead_ii_nel);
			xm_dim_inc_mask(&elidx, &blkdims, &mask_i);
		}
		xm_dim_inc_mask(&elidx, &blkdims, &mask_j);
	}
}

static void
unfold_kernel_memcpy(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	memcpy((char *)to + (j * stride + i) * size,
	    (const char *)from + offset * size,
	    lead_ii_nel * size);
}

static void
unfold_kernel_float(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	float *xto = to;
	const float *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[j * stride + i + k] = xfrom[offset];
		offset += size;
	}
}

static void
unfold_kernel_float_complex(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	float complex *xto = to;
	const float complex *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[j * stride + i + k] = xfrom[offset];
		offset += size;
	}
}

static void
unfold_kernel_double(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	double *xto = to;
	const double *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[j * stride + i + k] = xfrom[offset];
		offset += size;
	}
}

static void
unfold_kernel_double_complex(void *to, const void *from, size_t i, size_t j,
    size_t offset, size_t stride, size_t size, size_t lead_ii_nel)
{
	double complex *xto = to;
	const double complex *xfrom = from;
	size_t k;

	for (k = 0; k < lead_ii_nel; k++) {
		xto[j * stride + i + k] = xfrom[offset];
		offset += size;
	}
}

void
xm_tensor_unfold_block(const xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const void *from, void *to,
    size_t stride)
{
	kernel_fn_t kernel_fn;
	xm_dim_t blkdims, blkdimsp, elidx, idx, permutation;
	size_t ii, jj, kk, offset, inc, lead_ii, lead_ii_nel;
	size_t block_size_i, block_size_j, size;

	if (from == NULL || to == NULL || from == to)
		fatal("invalid argument");
	if (mask_i.n + mask_j.n != blkidx.n)
		fatal("invalid mask dimensions");

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	block_size_i = xm_dim_dot_mask(&blkdims, &mask_i);
	block_size_j = xm_dim_dot_mask(&blkdims, &mask_j);
	permutation = xm_tensor_get_block_permutation(tensor, blkidx);
	blkdimsp = xm_dim_permute(&blkdims, &permutation);
	elidx = xm_dim_zero(blkdims.n);

	inc = 1;
	lead_ii_nel = 1;

	if (mask_i.n > 0) {
		lead_ii = mask_i.i[0];
		for (kk = 0; kk < permutation.i[lead_ii]; kk++)
			inc *= blkdimsp.i[kk];
		for (ii = 0; ii < mask_i.n-1; ii++)
			mask_i.i[ii] = mask_i.i[ii+1];
		mask_i.n--;
		lead_ii_nel = blkdims.i[lead_ii];
	}
	if (inc == 1) {
		size = xm_scalar_sizeof(tensor->type);
		kernel_fn = unfold_kernel_memcpy;
	} else {
		size = inc;
		switch (tensor->type) {
		case XM_SCALAR_FLOAT:
			kernel_fn = unfold_kernel_float;
			break;
		case XM_SCALAR_FLOAT_COMPLEX:
			kernel_fn = unfold_kernel_float_complex;
			break;
		case XM_SCALAR_DOUBLE:
			kernel_fn = unfold_kernel_double;
			break;
		case XM_SCALAR_DOUBLE_COMPLEX:
			kernel_fn = unfold_kernel_double_complex;
			break;
		default:
			fatal("unexpected scalar type");
		}
	}
	for (jj = 0; jj < block_size_j; jj++) {
		xm_dim_zero_mask(&elidx, &mask_i);
		for (ii = 0; ii < block_size_i; ii += lead_ii_nel) {
			idx = xm_dim_permute(&elidx, &permutation);
			offset = xm_dim_offset(&idx, &blkdimsp);
			kernel_fn(to, from, ii, jj, offset, stride, size,
			    lead_ii_nel);
			xm_dim_inc_mask(&elidx, &blkdims, &mask_i);
		}
		xm_dim_inc_mask(&elidx, &blkdims, &mask_j);
	}
}

void
xm_tensor_free_block_data(xm_tensor_t *tensor)
{
	xm_dim_t idx, nblocks;
	uint64_t data_ptr;
	xm_block_type_t blocktype;

	nblocks = xm_tensor_get_nblocks(tensor);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		blocktype = xm_tensor_get_block_type(tensor, idx);
		if (blocktype == XM_BLOCK_TYPE_CANONICAL) {
			data_ptr = xm_tensor_get_block_data_ptr(tensor, idx);
			xm_allocator_deallocate(tensor->allocator, data_ptr);
		}
		xm_tensor_set_zero_block(tensor, idx);
		xm_dim_inc(&idx, &nblocks);
	}
}

void
xm_tensor_free(xm_tensor_t *tensor)
{
	if (tensor) {
		xm_block_space_free(tensor->bs);
		free(tensor->blocks);
		free(tensor);
	}
}
