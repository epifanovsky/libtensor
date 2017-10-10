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

#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include "alloc.h"
#include "blockspace.h"
#include "scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Block type. */
#define XM_BLOCK_TYPE_ZERO        0  /* see xm_tensor_set_zero_block */
#define XM_BLOCK_TYPE_CANONICAL   1  /* see xm_tensor_set_canonical_block */
#define XM_BLOCK_TYPE_DERIVATIVE  2  /* see xm_tensor_set_derivative_block */

/* Opaque tensor structure. */
typedef struct xm_tensor xm_tensor_t;

/* Create new block-tensor with all blocks set to zero-blocks. Type argument
 * specifies floating point type of a tensor and should be one of the
 * XM_SCALAR_ values. */
xm_tensor_t *xm_tensor_create(const xm_block_space_t *bs, int type,
    xm_allocator_t *allocator);

/* Create new block-tensor with all blocks set to newly allocated canonical
 * blocks. */
xm_tensor_t *xm_tensor_create_canonical(const xm_block_space_t *bs, int type,
    xm_allocator_t *allocator);

/* Create new block-tensor using block structure from "tensor". This function
 * only copies the block structure and does not copy actual data. */
xm_tensor_t *xm_tensor_create_structure(const xm_tensor_t *tensor, int type,
    xm_allocator_t *allocator);

/* Return block-space associated with this tensor. */
const xm_block_space_t *xm_tensor_get_block_space(const xm_tensor_t *tensor);

/* Return scalar type of this tensor. */
int xm_tensor_get_scalar_type(const xm_tensor_t *tensor);

/* Return allocator associated with this tensor. */
xm_allocator_t *xm_tensor_get_allocator(const xm_tensor_t *tensor);

/* Return absolute tensor dimensions in total number of elements. */
xm_dim_t xm_tensor_get_abs_dims(const xm_tensor_t *tensor);

/* Return tensor dimensions in number of blocks. */
xm_dim_t xm_tensor_get_nblocks(const xm_tensor_t *tensor);

/* Return an individual element of a tensor given its absolute index.
 * Note: this function is relatively slow. */
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx);

/* Return type of a block. This returns one of the XM_BLOCK_TYPE_ values. */
int xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return dimensions of a specific block. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return size in number of elements of a specific tensor block. */
size_t xm_tensor_get_block_size(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return size in number of bytes of a specific tensor block. */
size_t xm_tensor_get_block_bytes(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return size in number of elements of the largest tensor block. */
size_t xm_tensor_get_largest_block_size(const xm_tensor_t *tensor);

/* Return size in number of bytes of the largest tensor block. */
size_t xm_tensor_get_largest_block_bytes(const xm_tensor_t *tensor);

/* Return block data pointer. This returns XM_NULL_PTR for zero blocks. */
uint64_t xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Return permutation of a specific tensor block. */
xm_dim_t xm_tensor_get_block_permutation(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Return scalar factor of a specific tensor block. */
xm_scalar_t xm_tensor_get_block_scalar(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Set tensor block as zero-block (all elements of a block are zeros).
 * No actual data are stored for zero-blocks. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Set tensor block as canonical block allocating necessary data for storage.
 * Canonical blocks are the only ones that store actual data. Note: if blocks
 * are allocated using disk-backed allocator they should be at least several
 * megabytes in size for best performance. Use xm_block_space_autosplit to
 * split block-spaces into optimally-sized blocks. */
void xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Same as xm_tensor_set_canonical_block with the ability to specify
 * preallocated data pointer for storage.
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. Allocation must be large enough to hold block
 * data. */
void xm_tensor_set_canonical_block_raw(xm_tensor_t *tensor, xm_dim_t blkidx,
    uint64_t data_ptr);

/* Set tensor block as derivative block. A derivative block is a copy of some
 * canonical block with applied permutation and multiplication by a scalar
 * factor. No actual data are stored for derivative blocks. */
void xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar);

/* Create a list of all canonical blocks of this tensor. The memory used by the
 * list should be released using free() function when it is no longer needed.
 * The number of elements in the blklist array will be stored in the nblklist
 * variable. */
void xm_tensor_get_canonical_block_list(const xm_tensor_t *tensor,
    xm_dim_t **blklist, size_t *nblklist);

/* Read tensor block data into memory buffer. The buffer has to be large enough
 * to hold the data. */
void xm_tensor_read_block(const xm_tensor_t *tensor, xm_dim_t blkidx,
    void *buf);

/* Write tensor block data from memory buffer. */
void xm_tensor_write_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    const void *buf);

/* Unfold block into the matrix form. The sequences of unfolding indices are
 * specified using the masks. The from parameter should point to the raw block
 * data in memory. The stride must be equal to or greater than the product of
 * mask_i block dimensions. */
void xm_tensor_unfold_block(const xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const void *from, void *to,
    size_t stride);

/* Fold block back from the matrix form. This is the inverse of the
 * xm_tensor_unfold_block function. On return, "to" will contain raw block data
 * that can be directly written to the block. Only canonical blocks can be
 * folded. */
void xm_tensor_fold_block(const xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const void *from, void *to,
    size_t stride);

/* Deallocate associated data for all blocks of this tensor. This resets all
 * blocks to zero. */
void xm_tensor_free_block_data(xm_tensor_t *tensor);

/* Release resources associated with a tensor. The actual block data are not
 * freed by this function. Use xm_tensor_free_block_data to do it. */
void xm_tensor_free(xm_tensor_t *tensor);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TENSOR_H_INCLUDED */
