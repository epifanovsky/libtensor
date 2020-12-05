/*
 * Copyright (c) 2017-2018 Ilya Kaliman
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

#ifndef XM_SCALAR_H_INCLUDED
#define XM_SCALAR_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

/** \file
 *  \brief Operations on vectors of scalar values. */

#ifdef __cplusplus
extern "C" {
#endif

/** Underlying type of scalar values. */
typedef enum {
	/** Scalar is float. */
	XM_SCALAR_FLOAT = 0,
	/** Scalar is float complex. */
	XM_SCALAR_FLOAT_COMPLEX,
	/** Scalar is double. */
	XM_SCALAR_DOUBLE,
	/** Scalar is double complex. */
	XM_SCALAR_DOUBLE_COMPLEX,
} xm_scalar_type_t;

/** Scalar type. It is defined as the largest floating point type convertible
 *  to all other types. */
#ifdef __cplusplus
typedef std::complex<double> xm_scalar_t;
#else
typedef double complex xm_scalar_t;
#endif

/** Check validity of scalar type.
 *  \param type Scalar type.
 *  \return Zero if the scalar type is invalid. */
int xm_scalar_check_type(xm_scalar_type_t type);

/** Return size of the scalar type in bytes.
 *  \param type One of ::xm_scalar_type_t values.
 *  \return Size of the scalar type in bytes. */
size_t xm_scalar_sizeof(xm_scalar_type_t type);

/** Perform addition of two xm_scalar_t values.
 *  \param a First value.
 *  \param b Second value.
 *  \param type Scalar type.
 *  \return Sum of a and b. */
xm_scalar_t xm_scalar_add(xm_scalar_t a, xm_scalar_t b, xm_scalar_type_t type);

/** Perform multiplication of two xm_scalar_t values.
 *  \param a First value.
 *  \param b Second value.
 *  \param type Scalar type.
 *  \return Product of a and b. */
xm_scalar_t xm_scalar_mul(xm_scalar_t a, xm_scalar_t b, xm_scalar_type_t type);

/** Return element of a vector.
 *  \param x Data vector.
 *  \param idx Zero-based element index.
 *  \param type Scalar type.
 *  \return Element of a vector at a specified index. */
xm_scalar_t xm_scalar_get_element(void *x, size_t idx, xm_scalar_type_t type);

/** Set all elements of a vector to same value.
 *  \param x Data vector.
 *  \param a Scalar value.
 *  \param len Length of vector \p x in number of elements.
 *  \param type Scalar type. */
void xm_scalar_set(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type);

/** Multiply all elements of a vector by a scalar factor.
 *  \param x Data vector.
 *  \param a Scalar factor.
 *  \param len Length of vector \p x in number of elements.
 *  \param type Scalar type. */
void xm_scalar_scale(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type);

/** Perform vector addition x = a * x + b * y.
 *  \param x Vector x.
 *  \param a Scalar value a.
 *  \param y Vector y.
 *  \param b Scalar value b.
 *  \param len Length of vector \p x in number of elements.
 *  \param type Scalar type. */
void xm_scalar_axpy(void *x, xm_scalar_t a, const void *y, xm_scalar_t b,
    size_t len, xm_scalar_type_t type);

/** Perform multiplication of vector elements: x = x * (a * y).
 *  \param x Vector x.
 *  \param a Scalar value a.
 *  \param y Vector y.
 *  \param len Length of vectors in number of elements.
 *  \param type Scalar type. */
void xm_scalar_vec_mul(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type);

/** Perform division of vector elements: x = x / (a * y).
 *  \param x Vector x.
 *  \param a Scalar value a.
 *  \param y Vector y.
 *  \param len Length of vectors in number of elements.
 *  \param type Scalar type. */
void xm_scalar_vec_div(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type);

/** Compute dot product of two vectors.
 *  \param x First vector.
 *  \param y Second vector.
 *  \param len Length of vectors in number of elements.
 *  \param type Scalar type.
 *  \return Dot product of two vectors. */
xm_scalar_t xm_scalar_dot(const void *x, const void *y, size_t len,
    xm_scalar_type_t type);

/** Convert data from one scalar type to another.
 *  \param x Destination vector.
 *  \param y Source vector.
 *  \param len Length of vectors in number of elements.
 *  \param xtype Destination scalar type.
 *  \param ytype Source scalar type. */
void xm_scalar_convert(void *x, const void *y, size_t len,
    xm_scalar_type_t xtype, xm_scalar_type_t ytype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_SCALAR_H_INCLUDED */
