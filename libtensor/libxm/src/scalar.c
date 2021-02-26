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

#include <string.h>

#include "scalar.h"
#include "util.h"

int
xm_scalar_check_type(xm_scalar_type_t type)
{
	return (type == XM_SCALAR_FLOAT ||
		type == XM_SCALAR_FLOAT_COMPLEX ||
		type == XM_SCALAR_DOUBLE ||
		type == XM_SCALAR_DOUBLE_COMPLEX);
}

size_t
xm_scalar_sizeof(xm_scalar_type_t type)
{
	static const size_t tbl[] = {
		sizeof(float),
		sizeof(float complex),
		sizeof(double),
		sizeof(double complex),
	};
	if (!xm_scalar_check_type(type))
		fatal("unexpected scalar type");
	return tbl[type];
}

xm_scalar_t
xm_scalar_add(xm_scalar_t a, xm_scalar_t b, xm_scalar_type_t type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float x = (float)a + (float)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex x = (float complex)a + (float complex)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_DOUBLE: {
		double x = (double)a + (double)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex x = (double complex)a + (double complex)b;
		return (xm_scalar_t)x;
	}
	default:
		fatal("unexpected scalar type");
	}
	return 0;
}

xm_scalar_t
xm_scalar_mul(xm_scalar_t a, xm_scalar_t b, xm_scalar_type_t type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float x = (float)a * (float)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex x = (float complex)a * (float complex)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_DOUBLE: {
		double x = (double)a * (double)b;
		return (xm_scalar_t)x;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex x = (double complex)a * (double complex)b;
		return (xm_scalar_t)x;
	}
	default:
		fatal("unexpected scalar type");
	}
	return 0;
}

xm_scalar_t
xm_scalar_get_element(void *x, size_t idx, xm_scalar_type_t type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *y = x;
		return (xm_scalar_t)y[idx];
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *y = x;
		return (xm_scalar_t)y[idx];
	}
	case XM_SCALAR_DOUBLE: {
		double *y = x;
		return (xm_scalar_t)y[idx];
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *y = x;
		return (xm_scalar_t)y[idx];
	}
	default:
		fatal("unexpected scalar type");
	}
	return 0;
}

void
xm_scalar_set(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = (float)a;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = (float complex)a;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = (double)a;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = (double complex)a;
		return;
	}
	default:
		fatal("unexpected scalar type");
	}
}

void
xm_scalar_scale(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= (float)a;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= (float complex)a;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= (double)a;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= (double complex)a;
		return;
	}
	default:
		fatal("unexpected scalar type");
	}
}

void
xm_scalar_axpy(void *x, xm_scalar_t a, const void *y, xm_scalar_t b, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		float aa = (float)a;
		float bb = (float)b;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] + bb * yy[i];
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		float complex aa = (float complex)a;
		float complex bb = (float complex)b;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] + bb * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		double aa = (double)a;
		double bb = (double)b;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] + bb * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		double complex aa = (double complex)a;
		double complex bb = (double complex)b;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] + bb * yy[i];
		return;
	}
	default:
		fatal("unexpected scalar type");
	}
}

void
xm_scalar_vec_mul(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		float aa = (float)a;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] * yy[i];
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		float complex aa = (float complex)a;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		double aa = (double)a;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		double complex aa = (double complex)a;
		for (i = 0; i < len; i++)
			xx[i] = aa * xx[i] * yy[i];
		return;
	}
	default:
		fatal("unexpected scalar type");
	}
}

void
xm_scalar_vec_div(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		float aa = (float)a;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (aa * yy[i]);
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		float complex aa = (float complex)a;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (aa * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		double aa = (double)a;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (aa * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		double complex aa = (double complex)a;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (aa * yy[i]);
		return;
	}
	default:
		fatal("unexpected scalar type");
	}
}

xm_scalar_t
xm_scalar_dot(const void *x, const void *y, size_t len, xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		const float *xx = x;
		const float *yy = y;
		float dot = 0.0;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return (xm_scalar_t)dot;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		const float complex *xx = x;
		const float complex *yy = y;
		float complex dot = 0.0;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return (xm_scalar_t)dot;
	}
	case XM_SCALAR_DOUBLE: {
		const double *xx = x;
		const double *yy = y;
		double dot = 0.0;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return (xm_scalar_t)dot;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		const double complex *xx = x;
		const double complex *yy = y;
		double complex dot = 0.0;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return (xm_scalar_t)dot;
	}
	default:
		fatal("unexpected scalar type");
	}
	return 0;
}

void
xm_scalar_convert(void *x, const void *y, size_t len, xm_scalar_type_t xtype,
    xm_scalar_type_t ytype)
{
	size_t i;

	if (xtype == ytype) {
		memcpy(x, y, len * xm_scalar_sizeof(xtype));
		return;
	}
	if (xtype == XM_SCALAR_DOUBLE && ytype == XM_SCALAR_FLOAT) {
		double *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = (double)yy[i];
		return;
	}
	if (xtype == XM_SCALAR_FLOAT && ytype == XM_SCALAR_DOUBLE) {
		float *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = (float)yy[i];
		return;
	}
	if (xtype == XM_SCALAR_DOUBLE_COMPLEX &&
	    ytype == XM_SCALAR_FLOAT_COMPLEX) {
		double complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = (double complex)yy[i];
		return;
	}
	if (xtype == XM_SCALAR_FLOAT_COMPLEX &&
	    ytype == XM_SCALAR_DOUBLE_COMPLEX) {
		float complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = (float complex)yy[i];
		return;
	}
	fatal("unsupported scalar conversion");
}
