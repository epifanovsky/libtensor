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

#include <assert.h>

#include "scalar.h"

size_t
xm_scalar_sizeof(int type)
{
	static const size_t tbl[] = {
		sizeof(float),
		sizeof(float complex),
		sizeof(double),
		sizeof(double complex),
	};

	assert(type >= 0 && type < 4);

	return tbl[type];
}

void
xm_scalar_set(void *buf, size_t len, int type, xm_scalar_t x)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] = x;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] = x;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] = x;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] = x;
		return;
	}
	}
}

void
xm_scalar_mul(void *buf, size_t len, int type, xm_scalar_t x)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] *= x;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] *= x;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] *= x;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xbuf = buf;
		for (i = 0; i < len; i++)
			xbuf[i] *= x;
		return;
	}
	}
}

void
xm_scalar_axpy(xm_scalar_t a, void *x, const void *y, size_t len, int type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + yy[i];
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + yy[i];
		return;
	}
	}
}
