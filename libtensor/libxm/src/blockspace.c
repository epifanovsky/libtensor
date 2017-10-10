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

#include "blockspace.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define XM_AUTOSPLIT_BLOCK_SIZE 32

struct xm_block_space {
	xm_dim_t dims, nblocks;
	size_t *splits[XM_MAX_DIM];
};

xm_block_space_t *
xm_block_space_create(xm_dim_t dims)
{
	xm_block_space_t *ret;
	size_t i;

	assert(dims.n > 0 && dims.n <= XM_MAX_DIM);

	if ((ret = calloc(1, sizeof *ret)) == NULL)
		return NULL;
	ret->dims = dims;
	ret->nblocks = xm_dim_same(ret->dims.n, 1);
	for (i = 0; i < ret->dims.n; i++) {
		if ((ret->splits[i] = malloc(2 * sizeof(size_t))) == NULL) {
			xm_block_space_free(ret);
			return NULL;
		}
		ret->splits[i][0] = 0;
		ret->splits[i][1] = ret->dims.i[i];
	}
	return ret;
}

xm_block_space_t *
xm_block_space_clone(const xm_block_space_t *bs)
{
	xm_block_space_t *ret;
	size_t i;

	if ((ret = calloc(1, sizeof *ret)) == NULL)
		return NULL;
	ret->dims = bs->dims;
	ret->nblocks = bs->nblocks;
	for (i = 0; i < ret->dims.n; i++) {
		ret->splits[i] = malloc((ret->nblocks.i[i]+1)*sizeof(size_t));
		if (ret->splits[i] == NULL) {
			xm_block_space_free(ret);
			return NULL;
		}
		memcpy(ret->splits[i], bs->splits[i],
		    (ret->nblocks.i[i]+1)*sizeof(size_t));
	}
	return ret;
}

xm_block_space_t *
xm_block_space_permute_clone(const xm_block_space_t *bs, xm_dim_t permutation)
{
	xm_block_space_t *ret;
	size_t i, j;

	if ((ret = calloc(1, sizeof *ret)) == NULL)
		return NULL;
	ret->dims = xm_dim_permute(&bs->dims, &permutation);
	ret->nblocks = xm_dim_permute(&bs->nblocks, &permutation);
	for (j = 0; j < ret->dims.n; j++) {
		i = permutation.i[j];
		assert(ret->nblocks.i[i] == bs->nblocks.i[j]);
		ret->splits[i] = malloc((ret->nblocks.i[i]+1)*sizeof(size_t));
		if (ret->splits[i] == NULL) {
			xm_block_space_free(ret);
			return NULL;
		}
		memcpy(ret->splits[i], bs->splits[j],
		    (ret->nblocks.i[i]+1)*sizeof(size_t));
	}
	return ret;
}

size_t
xm_block_space_get_ndims(const xm_block_space_t *bs)
{
	return bs->dims.n;
}

xm_dim_t
xm_block_space_get_abs_dims(const xm_block_space_t *bs)
{
	return bs->dims;
}

xm_dim_t
xm_block_space_get_nblocks(const xm_block_space_t *bs)
{
	return bs->nblocks;
}

void
xm_block_space_split(xm_block_space_t *bs, size_t dim, size_t x)
{
	size_t i, *nptr;

	assert(dim < bs->dims.n);
	assert(x <= bs->dims.i[dim]);

	for (i = 0; i < bs->nblocks.i[dim]+1; i++) {
		if (bs->splits[dim][i] > x) {
			if ((nptr = realloc(bs->splits[dim],
			    (bs->nblocks.i[dim]+2)*sizeof(size_t))) == NULL)
				return;
			bs->splits[dim] = nptr;
			break;
		} else if (bs->splits[dim][i] == x)
			return;
	}
	memmove(bs->splits[dim]+i+1, bs->splits[dim]+i,
	    (bs->nblocks.i[dim]-i+1)*sizeof(size_t));
	bs->splits[dim][i] = x;
	bs->nblocks.i[dim]++;
}

void
xm_block_space_autosplit(xm_block_space_t *bs)
{
	const size_t blksz = XM_AUTOSPLIT_BLOCK_SIZE;
	size_t i, j, pos;

	for (j = 0; j < bs->dims.n; j++) {
		size_t dim = bs->dims.i[j];
		size_t nblks = dim % blksz ? dim / blksz + 1 : dim / blksz;

		for(i = 0, pos = 0; i < nblks - 1; i++) {
			size_t sz = dim / (nblks - i);

			if(sz > 1 && sz % 2 && nblks - i > 1) {
				if(sz < blksz) sz++;
				else sz--;
			}
			dim -= sz;
			pos += sz;
			xm_block_space_split(bs, j, pos);
		}
	}
}

size_t
xm_block_space_get_split(xm_block_space_t *bs, size_t dim, size_t i)
{
	assert(dim < bs->nblocks.n);
	assert(i <= bs->nblocks.i[dim]);

	return bs->splits[dim][i];
}

xm_dim_t
xm_block_space_get_block_dims(const xm_block_space_t *bs, xm_dim_t blkidx)
{
	xm_dim_t blkdims;
	size_t i;

	assert(bs->dims.n == blkidx.n);
	assert(xm_dim_less(&blkidx, &bs->nblocks));

	blkdims.n = bs->dims.n;
	for (i = 0; i < blkdims.n; i++) {
		blkdims.i[i] = bs->splits[i][blkidx.i[i]+1] -
		    bs->splits[i][blkidx.i[i]];
	}
	return blkdims;
}

size_t
xm_block_space_get_block_size(const xm_block_space_t *bs, xm_dim_t blkidx)
{
	size_t i, size = 1;

	assert(bs->dims.n == blkidx.n);
	assert(xm_dim_less(&blkidx, &bs->nblocks));

	for (i = 0; i < bs->dims.n; i++) {
		size *= bs->splits[i][blkidx.i[i]+1] -
		    bs->splits[i][blkidx.i[i]];
	}
	return size;
}

size_t
xm_block_space_get_largest_block_size(const xm_block_space_t *bs)
{
	size_t i, j, size, maxsize, blksize = 1;

	for (i = 0; i < bs->dims.n; i++) {
		maxsize = 0;
		for (j = 0; j < bs->nblocks.i[i]; j++) {
			size = bs->splits[i][j+1]-bs->splits[i][j];
			if (size > maxsize)
				maxsize = size;
		}
		blksize *= maxsize;
	}
	return blksize;
}

void
xm_block_space_decompose_index(const xm_block_space_t *bs, xm_dim_t idx,
    xm_dim_t *blkidx, xm_dim_t *elidx)
{
	size_t i, j;

	assert(bs->dims.n == idx.n);

	*blkidx = xm_dim_zero(bs->dims.n);
	*elidx = xm_dim_zero(bs->dims.n);
	for (i = 0; i < bs->dims.n; i++)
		for (j = 0; j < bs->nblocks.i[i]; j++)
			if (bs->splits[i][j+1] > idx.i[i]) {
				blkidx->i[i] = j;
				elidx->i[i] = idx.i[i] - bs->splits[i][j];
				break;
			}
}

int
xm_block_space_eq(const xm_block_space_t *bsa, const xm_block_space_t *bsb)
{
	size_t i;

	if (bsa->dims.n != bsb->dims.n)
		return 0;
	for (i = 0; i < bsa->dims.n; i++)
		if (!xm_block_space_eq1(bsa, i, bsb, i))
			return 0;
	return 1;
}

int
xm_block_space_eq1(const xm_block_space_t *bsa, size_t dima,
    const xm_block_space_t *bsb, size_t dimb)
{
	size_t i;

	assert(dima < bsa->nblocks.n);
	assert(dimb < bsb->nblocks.n);

	if (bsa->nblocks.i[dima] != bsb->nblocks.i[dimb])
		return 0;
	for (i = 0; i < bsa->nblocks.i[dima]+1; i++)
		if (bsa->splits[dima][i] != bsb->splits[dimb][i])
			return 0;
	return 1;
}

void
xm_block_space_free(xm_block_space_t *bs)
{
	size_t i;

	if (bs) {
		for (i = 0; i < bs->dims.n; i++)
			free(bs->splits[i]);
		free(bs);
	}
}
