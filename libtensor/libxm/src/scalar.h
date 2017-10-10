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

#ifndef XM_SCALAR_H_INCLUDED
#define XM_SCALAR_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define XM_SCALAR_FLOAT           0
#define XM_SCALAR_FLOAT_COMPLEX   1
#define XM_SCALAR_DOUBLE          2
#define XM_SCALAR_DOUBLE_COMPLEX  3

/* Largest floating point type convertible to all other types. */
#ifdef __cplusplus
typedef std::complex<double> xm_scalar_t;
#else
typedef double complex xm_scalar_t;
#endif

size_t xm_scalar_sizeof(int type);
void xm_scalar_set(void *buf, size_t len, int type, xm_scalar_t x);
void xm_scalar_mul(void *buf, size_t len, int type, xm_scalar_t x);
void xm_scalar_axpy(xm_scalar_t a, void *x, const void *y, size_t len,
    int type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_SCALAR_H_INCLUDED */
