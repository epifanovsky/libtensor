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

/** \file
 *  \brief Operations on multidimensional indices. */

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of dimensions. */
#define XM_MAX_DIM 8

/** Multidimensional index. */
typedef struct {
	/** Number of dimensions. Must not be greater than ::XM_MAX_DIM. */
	size_t n;
	/** Dimension sizes. Only the first \p n values are used. */
	size_t i[XM_MAX_DIM];
} xm_dim_t;

/** Return new ::xm_dim_t with all indices initialized to zero.
 *  \param ndim Number of dimensions of the new dim.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_zero(size_t ndim);

/** Return new ::xm_dim_t with all indices initialized to the same value.
 *  \param ndim Number of dimensions of the new dim.
 *  \param dim Size of the dimensions.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_same(size_t ndim, size_t dim);

/** Initialize a 1-D dim.
 *  \param dim1 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_1(size_t dim1);

/** Initialize a 2-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_2(size_t dim1, size_t dim2);

/** Initialize a 3-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_3(size_t dim1, size_t dim2, size_t dim3);

/** Initialize a 4-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \param dim4 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4);

/** Initialize a 5-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \param dim4 Dimension size.
 *  \param dim5 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5);

/** Initialize a 6-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \param dim4 Dimension size.
 *  \param dim5 Dimension size.
 *  \param dim6 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6);

/** Initialize a 7-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \param dim4 Dimension size.
 *  \param dim5 Dimension size.
 *  \param dim6 Dimension size.
 *  \param dim7 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_7(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7);

/** Initialize a 8-D dim.
 *  \param dim1 Dimension size.
 *  \param dim2 Dimension size.
 *  \param dim3 Dimension size.
 *  \param dim4 Dimension size.
 *  \param dim5 Dimension size.
 *  \param dim6 Dimension size.
 *  \param dim7 Dimension size.
 *  \param dim8 Dimension size.
 *  \return New instance of ::xm_dim_t. */
xm_dim_t xm_dim_8(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6, size_t dim7, size_t dim8);

/** Test equality of two dims.
 *  \param a First dim.
 *  \param b Second dim.
 *  \return Non-zero if the dims are equal. */
int xm_dim_eq(const xm_dim_t *a, const xm_dim_t *b);

/** Test inequality of two dims.
 *  \param a First dim.
 *  \param b Second dim.
 *  \return Non-zero if the dims are NOT equal. */
int xm_dim_ne(const xm_dim_t *a, const xm_dim_t *b);

/** Return non-zero if an index is within zero-index and \p dim.
 *  \param idx An index to test.
 *  \param dim Maximum dimensions.
 *  \return Non-zero if the index is within zero-index and \p dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/** Return product of the indices of a \p dim.
 *  \param dim A dim.
 *  \return Product of the indices of a \p dim. */
size_t xm_dim_dot(const xm_dim_t *dim);

/** Return absolute offset of an index.
 *  \param idx An index within \p dim.
 *  \param dim A dim.
 *  \return Absolute offset of an index. */
size_t xm_dim_offset(const xm_dim_t *idx, const xm_dim_t *dim);

/** Compute an index from an offset.
 *  \param offset An absolute offset of an index.
 *  \param dim A dim.
 *  \return An index with the specified absolute offset. */
xm_dim_t xm_dim_from_offset(size_t offset, const xm_dim_t *dim);

/** Increment an index by one wrapping on dimensions. Can be used to iterate
 *  over all elements of a dim.
 *  \param idx An index.
 *  \param dim Maximum dimensions. */
void xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);

/** Set to zero all indices of a dim according to the mask.
 *  \param dim A dim.
 *  \param mask Mask specifying affected indices. */
void xm_dim_zero_mask(xm_dim_t *dim, const xm_dim_t *mask);

/** Set \p maska indices of \p a to \p maskb indices of \p b.
 *  \param a Destination dim.
 *  \param maska Mask of \p a.
 *  \param b Source dim.
 *  \param maskb Mask of \p b. */
void xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *maska, const xm_dim_t *b,
    const xm_dim_t *maskb);

/** Return product of indices specified by mask.
 *  \param dim A dim.
 *  \param mask Mask specifying affected indices.
 *  \return Product of indices. */
size_t xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask);

/** Increment an index. This is similar to ::xm_dim_inc function, but affects
 *  only indices specified by \p mask.
 *  \param idx An index to increment.
 *  \param dim Maximum dimensions.
 *  \param mask Mask specifying affected indices. */
void xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask);

/** Return an identity permutation.
 *  \param ndim Dimensionality of the permutation.
 *  \return A new identity permutation. */
xm_dim_t xm_dim_identity_permutation(size_t ndim);

/** Return permutation of an index.
 *  \param idx An index.
 *  \param permutation Permutation.
 *  \return Index with elements permuted according to the \p permutation. */
xm_dim_t xm_dim_permute(const xm_dim_t *idx, const xm_dim_t *permutation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_DIM_H_INCLUDED */
