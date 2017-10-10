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

#ifndef XM_DIM_H_INCLUDED
#define XM_DIM_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of dimensions. */
#define XM_MAX_DIM 8

/* Multidimensional index. */
typedef struct {
	size_t n, i[XM_MAX_DIM];
} xm_dim_t;

/* Return a dim with all indices initialized to zero. */
xm_dim_t xm_dim_zero(size_t ndim);

/* Return a dim with all indices initialized to the same value. */
xm_dim_t xm_dim_same(size_t ndim, size_t dim);

/* Initialize a 1-D dim. */
xm_dim_t xm_dim_1(size_t dim1);

/* Initialize a 2-D dim. */
xm_dim_t xm_dim_2(size_t dim1, size_t dim2);

/* Initialize a 3-D dim. */
xm_dim_t xm_dim_3(size_t dim1, size_t dim2, size_t dim3);

/* Initialize a 4-D dim. */
xm_dim_t xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4);

/* Initialize a 5-D dim. */
xm_dim_t xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5);

/* Initialize a 6-D dim. */
xm_dim_t xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6);

/* Initialize a 7-D dim. */
xm_dim_t xm_dim_7(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7);

/* Initialize a 8-D dim. */
xm_dim_t xm_dim_8(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7, size_t dim8);

/* Return non-zero if two dims are equal. */
int xm_dim_eq(const xm_dim_t *a, const xm_dim_t *b);

/* Return non-zero if two dims are not equal. */
int xm_dim_ne(const xm_dim_t *a, const xm_dim_t *b);

/* Return non-zero if an index is within zero and dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/* Return product of all indices of a dim. */
size_t xm_dim_dot(const xm_dim_t *dim);

/* Return absolute offset of an index in a dim. */
size_t xm_dim_offset(const xm_dim_t *idx, const xm_dim_t *dim);

/* Compute an index from an offset in the specified dim. */
xm_dim_t xm_dim_from_offset(size_t offset, const xm_dim_t *dim);

/* Increment an index by one wrapping on dimensions. Can be used to iterate
 * over all elements of a dim. */
void xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);

/* Set to zero all indices of this dim specified by mask. */
void xm_dim_zero_mask(xm_dim_t *dim, const xm_dim_t *mask);

/* Set maska indices of "a" to maskb indices of "b". */
void xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *maska, const xm_dim_t *b,
    const xm_dim_t *maskb);

/* Return product of indices specified by mask. */
size_t xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask);

/* Increment an index. This is similar to xm_dim_inc, but affects only indices
 * specified by mask. */
void xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask);

/* Return n-dimensional identity permutation. */
xm_dim_t xm_dim_identity_permutation(size_t ndim);

/* Return a specific permutation of an index. */
xm_dim_t xm_dim_permute(const xm_dim_t *idx, const xm_dim_t *permutation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_DIM_H_INCLUDED */
