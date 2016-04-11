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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aux.h"

xm_scalar_t
xm_random_scalar(void)
{
	double a = drand48() - 0.5;
#if defined(XM_SCALAR_DOUBLE_COMPLEX) || defined(XM_SCALAR_FLOAT_COMPLEX)
	double b = drand48() - 0.5;
	return ((xm_scalar_t)(a + b * I));
#else
	return ((xm_scalar_t)a);
#endif
}

static uintptr_t
xm_allocate_new_block(struct xm_allocator *allocator, const xm_dim_t *dim,
    int type)
{
	uintptr_t ptr;
	size_t size, size_bytes, i;
	xm_scalar_t *data;

	size = xm_dim_dot(dim);
	size_bytes = size * sizeof(xm_scalar_t);

	ptr = xm_allocator_allocate(allocator, size_bytes);
	if (ptr == XM_NULL_PTR)
		return (XM_NULL_PTR);
	if (type == XM_INIT_NONE)
		return (ptr);
	if (type == XM_INIT_RAND) {
		if ((data = malloc(size_bytes)) == NULL) {
			perror("malloc");
			abort();
		}
		for (i = 0; i < size; i++) {
			data[i] = xm_random_scalar();
		}
		xm_allocator_write(allocator, ptr, data, size_bytes);
		free(data);
		return (ptr);
	}
	xm_allocator_memset(allocator, ptr, 0, size_bytes);
	return (ptr);
}

int
xm_tensor_init(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx;
	size_t i, size;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	idx = xm_dim_zero(dim.n);
	blk_dim = xm_dim_same(dim.n, block_size);
	size = xm_dim_dot(&dim);

	for (i = 0; i < size; i++) {
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);
		xm_dim_inc(&idx, &dim);
	}

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_oo(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 2);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);
	} }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_ov(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 2);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);
	} }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_vv(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 2);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);
	} }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_vvx(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 3);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);
	} } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_oooo(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_ooov(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_oovv(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_ovov(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_ovvv(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_vvvv(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);
	} } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_ooovvv(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t blk_dim, dim, idx, idx2, perm, perm2;
	time_t wall;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 6);

	blk_dim = xm_dim_same(dim.n, block_size);
	idx = xm_dim_zero(dim.n);
	perm = xm_dim_identity_permutation(dim.n);

	for (idx.i[0] = 0; idx.i[0] < dim.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < dim.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < dim.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < dim.i[3]; idx.i[3]++) {
	for (idx.i[4] = 0; idx.i[4] < dim.i[4]; idx.i[4]++) {
	for (idx.i[5] = 0; idx.i[5] < dim.i[5]; idx.i[5]++) {
		if (xm_tensor_get_block_data_ptr(tensor, &idx) != XM_NULL_PTR)
			continue;
		block = xm_allocate_new_block(allocator, &blk_dim, type);
		if (block == XM_NULL_PTR)
			return (XM_RESULT_NO_MEMORY);
		xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[4] = idx.i[5];
		idx2.i[5] = idx.i[4];
		perm2 = perm;
		perm2.i[4] = perm.i[5];
		perm2.i[5] = perm.i[4];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[4] = idx.i[5];
		idx2.i[5] = idx.i[4];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[4] = perm.i[5];
		perm2.i[5] = perm.i[4];
		if (xm_tensor_get_block_data_ptr(tensor, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(tensor, &idx2, &idx, &perm2, 1.0);
	} } } } } }

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_13(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t dim, blk_dim, idx, idx2, perm;
	time_t wall;

	(void)block_size;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 2);

	blk_dim = xm_dim_2(2, 2);
	idx = xm_dim_2(0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 3);
	idx = xm_dim_2(1, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(2, 3);
	idx = xm_dim_2(0, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	idx = xm_dim_2(1, 0);
	idx2 = xm_dim_2(0, 1);
	perm = xm_dim_2(1, 0);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, 1.0);

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_13c(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t dim, blk_dim, idx;
	time_t wall;

	(void)block_size;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 2);

	blk_dim = xm_dim_2(2, 2);
	idx = xm_dim_2(0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 3);
	idx = xm_dim_2(1, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(2, 3);
	idx = xm_dim_2(0, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 2);
	idx = xm_dim_2(1, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_14(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t dim, blk_dim, idx, idx2, perm;
	time_t wall;

	(void)block_size;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_4(2, 2, 2, 2);
	idx = xm_dim_4(0, 0, 0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(4, 2, 2, 2);
	idx = xm_dim_4(1, 0, 0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 1, 0, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 2, 2);
	idx = xm_dim_4(1, 1, 0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 0, 1, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, 1.0);

	blk_dim = xm_dim_4(4, 2, 4, 2);
	idx = xm_dim_4(1, 0, 1, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 1, 1, 0);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 4, 2);
	idx = xm_dim_4(1, 1, 1, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 1, 0);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(1, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(0, 1, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 3, 2);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(1, 1, 0, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(0, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(1, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(0, 1, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(3, 2, 0, 1);
	xm_tensor_set_block(tensor, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 4, 4);
	idx = xm_dim_4(1, 1, 1, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}

int
xm_tensor_init_14b(struct xm_tensor *tensor, struct xm_allocator *allocator,
    size_t block_size, int type)
{
	uintptr_t block;
	xm_dim_t dim, blk_dim, idx;
	time_t wall;

	(void)block_size;

	wall = time(NULL);
	fprintf(stderr, "%s(%s)\n", __func__, xm_tensor_get_label(tensor));

	dim = xm_tensor_get_dim(tensor);
	assert(dim.n == 4);

	blk_dim = xm_dim_4(3, 3, 2, 2);
	idx = xm_dim_4(0, 0, 0, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 2, 4);
	idx = xm_dim_4(0, 0, 0, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 4, 2);
	idx = xm_dim_4(0, 0, 1, 0);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 4, 4);
	idx = xm_dim_4(0, 0, 1, 1);
	block = xm_allocate_new_block(allocator, &blk_dim, type);
	if (block == XM_NULL_PTR)
		return (XM_RESULT_NO_MEMORY);
	xm_tensor_set_source_block(tensor, &idx, &blk_dim, block);

	wall = time(NULL) - wall;
	fprintf(stderr, "%s done in %d sec\n", __func__, (int)wall);

	return (XM_RESULT_SUCCESS);
}
