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

#ifndef XM_BLOCKSPACE_H_INCLUDED
#define XM_BLOCKSPACE_H_INCLUDED

#include "dim.h"

/** \file
 *  \brief Operations on block-spaces. */

#ifdef __cplusplus
extern "C" {
#endif

/** Multidimensional block-space. */
typedef struct xm_block_space xm_block_space_t;

/** Create a block-space with specific absolute dimensions.
 *  \param dims Absolute dimensions of the new block-space.
 *  \return New block-space instance. */
xm_block_space_t *xm_block_space_create(xm_dim_t dims);

/** Create a deep copy of a block-space.
 *  \param bs Block-space.
 *  \return Deep copy of the block-space \p bs. */
xm_block_space_t *xm_block_space_clone(const xm_block_space_t *bs);

/** Create a deep copy of a block-space while applying permutation.
 *  \param bs Block-space.
 *  \param permutation Permutation to apply to the block-space.
 *  \return Deep copy of the block-space with permuted dimensions. */
xm_block_space_t *xm_block_space_permute_clone(const xm_block_space_t *bs,
    xm_dim_t permutation);

/** Return number of dimensions a block-space has.
 *  \param bs Block-space.
 *  \return Number of dimensions. */
size_t xm_block_space_get_ndims(const xm_block_space_t *bs);

/** Return absolute dimensions of a block-space.
 *  \param bs Block-space.
 *  \return Absolute dimensions of the block-space. */
xm_dim_t xm_block_space_get_abs_dims(const xm_block_space_t *bs);

/** Return block-space dimensions in number of blocks.
 *  \param bs Block-space.
 *  \return Block-space dimensions in blocks. */
xm_dim_t xm_block_space_get_nblocks(const xm_block_space_t *bs);

/** Split block-space along a dimension at a specific point.
 *  \param bs Block-space.
 *  \param dim Dimension index.
 *  \param x Split point along the dimension \p dim. */
void xm_block_space_split(xm_block_space_t *bs, size_t dim, size_t x);

/** Automatically split the block-space into optimally sized blocks.
 *  \param bs Block-space. */
void xm_block_space_autosplit(xm_block_space_t *bs);

/** Return split point position along the dimension \p dim.
 *  \param bs Block-space.
 *  \param dim Dimension index.
 *  \param i Index of the split point.
 *  \return Split point with index \p i. */
size_t xm_block_space_get_split(xm_block_space_t *bs, size_t dim, size_t i);

/** Return dimensions of the block with specific index.
 *  \param bs Block-space.
 *  \param blkidx Index of the block.
 *  \return Dimensions of the block. */
xm_dim_t xm_block_space_get_block_dims(const xm_block_space_t *bs,
    xm_dim_t blkidx);

/** Return size in number of elements of a specific block.
 *  \param bs Block-space.
 *  \param blkidx Index of the block.
 *  \return Size of the block in number of elements. */
size_t xm_block_space_get_block_size(const xm_block_space_t *bs,
    xm_dim_t blkidx);

/** Return size in number of elements of the largest block.
 *  \param bs Block-space.
 *  \return Largest block size in number of elements. */
size_t xm_block_space_get_largest_block_size(const xm_block_space_t *bs);

/** Decompose an absolute index into block index and element index inside
 *  the block.
 *  \param bs Block-space.
 *  \param idx Index to decompose.
 *  \param blkidx Output block index.
 *  \param elidx Output element index inside the block. */
void xm_block_space_decompose_index(const xm_block_space_t *bs, xm_dim_t idx,
    xm_dim_t *blkidx, xm_dim_t *elidx);

/** Compare block-spaces.
 *  \param bsa First block-space.
 *  \param bsb Second block-space.
 *  \return Non-zero if block-space structures are equal. */
int xm_block_space_eq(const xm_block_space_t *bsa, const xm_block_space_t *bsb);

/** Check if specific block-space dimensions have same block structures.
 *  \param bsa First block-space.
 *  \param dima Dimension index of the first block-space.
 *  \param bsb Second block-space.
 *  \param dimb Dimension index of the second block-space.
 *  \return Non-zero if the block-space dimensions are equal. */
int xm_block_space_eq1(const xm_block_space_t *bsa, size_t dima,
    const xm_block_space_t *bsb, size_t dimb);

/** Release resources used by the block-space.
 *  \param bs Pointer to the block-space. The pointer can be NULL. */
void xm_block_space_free(xm_block_space_t *bs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_BLOCKSPACE_H_INCLUDED */
