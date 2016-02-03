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

#ifndef XM_H_INCLUDED
#define XM_H_INCLUDED

#include "alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of tensor dimensions. */
#define XM_MAX_DIM                6

/* Result return codes. */
#define XM_RESULT_SUCCESS         0  /* Success. */
#define XM_RESULT_NO_MEMORY       1  /* Cannot allocate memory. */

#if defined(XM_SCALAR_FLOAT)
typedef float xm_scalar_t;
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#include <complex.h>
typedef double complex xm_scalar_t;
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#include <complex.h>
typedef float complex xm_scalar_t;
#else /* assume double */
typedef double xm_scalar_t;
#endif

/* Opaque tensor struct. */
struct xm_tensor;

/* Tensor index. */
typedef struct {
	size_t n, i[XM_MAX_DIM];
} xm_dim_t;

/* Returns libxm banner string. Print this in the program output. */
const char *xm_banner(void);

/* Initialize all indices of a dim to zero. */
xm_dim_t xm_dim_zero(size_t n);

/* Initialize all indices of a dim to the same value. */
xm_dim_t xm_dim_same(size_t n, size_t dim);

/* Initialize 2D dim. */
xm_dim_t xm_dim_2(size_t dim1, size_t dim2);

/* Initialize 3D dim. */
xm_dim_t xm_dim_3(size_t dim1, size_t dim2, size_t dim3);

/* Initialize 4D dim. */
xm_dim_t xm_dim_4(size_t dim1, size_t dim2, size_t dim3,
    size_t dim4);

/* Initialize 5D dim. */
xm_dim_t xm_dim_5(size_t dim1, size_t dim2, size_t dim3,
    size_t dim4, size_t dim5);

/* Initialize 6D dim. */
xm_dim_t xm_dim_6(size_t dim1, size_t dim2, size_t dim3,
    size_t dim4, size_t dim5, size_t dim6);

/* Returns an identity permutation. */
xm_dim_t xm_dim_identity_permutation(size_t n);

/* Returns dot product of all indices of a dim. */
size_t xm_dim_dot(const xm_dim_t *dim);

/* Returns non-zero if index is within zero and dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/* Increment an index by one wrapping on dimensions. */
size_t xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);

/* Set stream to log to. Setting stream to NULL disables logging. */
void xm_set_log_stream(FILE *stream);

/* Create a labeled tensor specifying its dimensions in blocks. */
struct xm_tensor *xm_tensor_create(struct xm_allocator *allocator,
    const xm_dim_t *dim, const char *label);

/* Copy tensor data. Tensors must have exactly the same block structure. */
void xm_tensor_copy_data(struct xm_tensor *dst, const struct xm_tensor *src);

/* Returns tensor label. */
const char *xm_tensor_get_label(const struct xm_tensor *tensor);

/* Returns tensor dimensions in blocks. */
xm_dim_t xm_tensor_get_dim(const struct xm_tensor *tensor);

/* Set tensor partition dimensions. */
void xm_tensor_set_part_dim(struct xm_tensor *tensor, const xm_dim_t *pdim);

/* Returns tensor partition dimensions. */
xm_dim_t xm_tensor_get_part_dim(const struct xm_tensor *tensor);

/* Returns tensor dimensions in number of elements. */
xm_dim_t xm_tensor_get_abs_dim(const struct xm_tensor *tensor);

/* Get tensor element given block index and element index within a block.
 * Note: this function is very slow. */
xm_scalar_t xm_tensor_get_element(struct xm_tensor *tensor,
    const xm_dim_t *blk_i, const xm_dim_t *el_i);

/* Get an element of a tensor given its absolute index.
 * Note: this function is very slow. */
xm_scalar_t xm_tensor_get_abs_element(struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Check if the block is non-zero. */
int xm_tensor_block_is_nonzero(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Check if the block is initialized. */
int xm_tensor_block_is_initialized(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Get block dimensions. */
xm_dim_t xm_tensor_get_block_dim(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Get block data pointer. */
uintptr_t xm_tensor_get_block_data_ptr(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Get permutation of a block. */
xm_dim_t xm_tensor_get_block_permutation(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Get scalar multiplier for a block. */
xm_scalar_t xm_tensor_get_block_scalar(const struct xm_tensor *tensor,
    const xm_dim_t *idx);

/* Set block to zero. */
void xm_tensor_set_zero_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim);

/* Set the source block. Each unique source block must be used once and must be
 * set using this function before being used in xm_tensor_set_block.
 * Note: if blocks are allocated using a disk-backed allocator they must
 * be at least several megabytes in size for best performance. */
void xm_tensor_set_source_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim, uintptr_t data);

/* Set a copy of a source block applying permutation and multiplying
 * by a scalar factor.
 * Note: if blocks are allocated using a disk-backed allocator they should
 * be at least several megabytes in size for best performance. */
void xm_tensor_set_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *source_idx, const xm_dim_t *perm, xm_scalar_t scalar);

/* Returns non-zero if all blocks of a tensor are initialized. */
int xm_tensor_is_initialized(const struct xm_tensor *tensor);

/* Release all resources associated with this tensor. */
void xm_tensor_free(struct xm_tensor *tensor);

/* Contract two tensors (c = alpha * a * b + beta * c) over contraction indices
 * specified by strings idxa and idxb. Permutation of tensor c is specified by
 * idxc. It is the user's responsibility to setup all tensors so that they have
 * correct symmetries and block-structures.
 *
 * Example: xm_contract(1.0, vvvv, oovv, 0.0, t2, "abcd", "ijcd", "ijab");
 */
int xm_contract(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    xm_scalar_t beta, struct xm_tensor *c, const char *idxa, const char *idxb,
    const char *idxc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
