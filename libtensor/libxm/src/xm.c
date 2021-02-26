/*
 * Copyright (c) 2014-2018 Ilya Kaliman
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef XM_USE_MPI
#include <mpi.h>
#endif

#include "xm.h"
#include "util.h"

void
xm_set(xm_tensor_t *a, xm_scalar_t x)
{
	xm_dim_t *blklist;
	xm_scalar_type_t scalartype;
	size_t i, maxblksize, nblklist;
	void *buf;
	int mpirank = 0, mpisize = 1;

#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	if ((buf = malloc(xm_tensor_get_largest_block_bytes(a))) == NULL)
		fatal("out of memory");
	maxblksize = xm_tensor_get_largest_block_size(a);
	scalartype = xm_tensor_get_scalar_type(a);
	xm_scalar_set(buf, x, maxblksize, scalartype);
	xm_tensor_get_canonical_block_list(a, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank)
			xm_tensor_write_block(a, blklist[i], buf);
	}
}
	free(buf);
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void
xm_copy(xm_tensor_t *a, xm_scalar_t s, const xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, *blklist;
	xm_scalar_type_t scalartypea, scalartypeb;
	size_t i, maxblkbytesa, maxblkbytesb, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(b))
		fatal("tensors must use same allocator");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	scalartypea = xm_tensor_get_scalar_type(a);
	scalartypeb = xm_tensor_get_scalar_type(b);
	zero = xm_dim_zero(0);
	maxblkbytesa = xm_tensor_get_largest_block_bytes(a);
	maxblkbytesb = xm_tensor_get_largest_block_bytes(b);
	xm_tensor_get_canonical_block_list(a, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_dim_t ia, ib;
	void *buf1a, *buf2a, *buf1b, *buf2b;
	size_t blksize;
	xm_block_type_t blocktype;

	if ((buf1a = malloc(maxblkbytesa)) == NULL)
		fatal("out of memory");
	if ((buf2a = malloc(maxblkbytesa)) == NULL)
		fatal("out of memory");
	if ((buf1b = malloc(maxblkbytesb)) == NULL)
		fatal("out of memory");
	if ((buf2b = malloc(maxblkbytesb)) == NULL)
		fatal("out of memory");
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank) {
			ia = blklist[i];
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			blksize = xm_tensor_get_block_size(b, ib);
			blocktype = xm_tensor_get_block_type(b, ib);
			if (s == 0 || blocktype == XM_BLOCK_TYPE_ZERO) {
				memset(buf2a, 0, maxblkbytesa);
			} else {
				xm_scalar_t scalar = xm_scalar_mul(s,
				    xm_tensor_get_block_scalar(b, ib),
				    scalartypeb);
				xm_tensor_read_block(b, ib, buf2b);
				xm_tensor_unfold_block(b, ib, cidxb, zero,
				    buf2b, buf1b, blksize);
				xm_scalar_scale(buf1b, scalar, blksize,
				    scalartypeb);
				xm_scalar_convert(buf1a, buf1b, blksize,
				    scalartypea, scalartypeb);
				xm_tensor_fold_block(a, ia, cidxa, zero, buf1a,
				    buf2a, blksize);
			}
			xm_tensor_write_block(a, ia, buf2a);
		}
	}
	free(buf1a);
	free(buf2a);
	free(buf1b);
	free(buf2b);
}
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void
xm_add(xm_scalar_t alpha, xm_tensor_t *a, xm_scalar_t beta,
    const xm_tensor_t *b, const char *idxa, const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, *blklist;
	xm_scalar_type_t scalartype;
	size_t i, maxblkbytes, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(b))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(b))
		fatal("tensors must have same scalar type");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	scalartype = xm_tensor_get_scalar_type(a);
	zero = xm_dim_zero(0);
	maxblkbytes = xm_tensor_get_largest_block_bytes(a);
	xm_tensor_get_canonical_block_list(a, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_dim_t ia, ib;
	void *buf1, *buf2;
	size_t blksize;
	xm_block_type_t blocktype;

	if ((buf1 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	if ((buf2 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank) {
			ia = blklist[i];
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			blksize = xm_tensor_get_block_size(b, ib);
			blocktype = xm_tensor_get_block_type(b, ib);
			if (beta == 0 || blocktype == XM_BLOCK_TYPE_ZERO) {
				memset(buf2, 0, maxblkbytes);
			} else {
				xm_scalar_t scalar = xm_scalar_mul(beta,
				    xm_tensor_get_block_scalar(b, ib),
				    scalartype);
				xm_tensor_read_block(b, ib, buf2);
				xm_tensor_unfold_block(b, ib, cidxb, zero, buf2,
				    buf1, blksize);
				xm_scalar_scale(buf1, scalar, blksize,
				    scalartype);
				xm_tensor_fold_block(a, ia, cidxa, zero, buf1,
				    buf2, blksize);
			}
			if (alpha == 0)
				xm_tensor_write_block(a, ia, buf2);
			else {
				xm_tensor_read_block(a, ia, buf1);
				xm_scalar_axpy(buf1, alpha, buf2, 1, blksize,
				    scalartype);
				xm_tensor_write_block(a, ia, buf1);
			}
		}
	}
	free(buf1);
	free(buf2);
}
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void
xm_mul(xm_tensor_t *a, const xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, *blklist;
	xm_scalar_type_t scalartype;
	size_t i, maxblkbytes, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(b))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(b))
		fatal("tensors must have same scalar type");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	scalartype = xm_tensor_get_scalar_type(a);
	zero = xm_dim_zero(0);
	maxblkbytes = xm_tensor_get_largest_block_bytes(a);
	xm_tensor_get_canonical_block_list(a, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_dim_t ia, ib;
	void *buf1, *buf2;
	size_t blksize;
	xm_block_type_t blocktype;

	if ((buf1 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	if ((buf2 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank) {
			xm_scalar_t scalar;
			ia = blklist[i];
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			blksize = xm_tensor_get_block_size(b, ib);
			blocktype = xm_tensor_get_block_type(b, ib);
			if (blocktype == XM_BLOCK_TYPE_ZERO) {
				memset(buf1, 0, maxblkbytes);
			} else {
				scalar = xm_tensor_get_block_scalar(b, ib);
				xm_tensor_read_block(b, ib, buf1);
				xm_tensor_unfold_block(b, ib, cidxb, zero, buf1,
				    buf2, blksize);
				xm_tensor_read_block(a, ia, buf1);
				xm_scalar_vec_mul(buf1, scalar, buf2, blksize,
				    scalartype);
			}
			xm_tensor_write_block(a, ia, buf1);
		}
	}
	free(buf1);
	free(buf2);
}
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void
xm_div(xm_tensor_t *a, const xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, *blklist;
	xm_scalar_type_t scalartype;
	size_t i, maxblkbytes, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(b))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(b))
		fatal("tensors must have same scalar type");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	scalartype = xm_tensor_get_scalar_type(a);
	zero = xm_dim_zero(0);
	maxblkbytes = xm_tensor_get_largest_block_bytes(a);
	xm_tensor_get_canonical_block_list(a, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_dim_t ia, ib;
	void *buf1, *buf2;
	size_t blksize;
	xm_block_type_t blocktype;

	if ((buf1 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	if ((buf2 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank) {
			xm_scalar_t scalar;
			ia = blklist[i];
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			blksize = xm_tensor_get_block_size(b, ib);
			blocktype = xm_tensor_get_block_type(b, ib);
			if (blocktype == XM_BLOCK_TYPE_ZERO)
				fatal("division by zero");
			scalar = xm_tensor_get_block_scalar(b, ib);
			xm_tensor_read_block(b, ib, buf1);
			xm_tensor_unfold_block(b, ib, cidxb, zero, buf1,
			    buf2, blksize);
			xm_tensor_read_block(a, ia, buf1);
			xm_scalar_vec_div(buf1, scalar, buf2, blksize,
			    scalartype);
			xm_tensor_write_block(a, ia, buf1);
		}
	}
	free(buf1);
	free(buf2);
}
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}

xm_scalar_t
xm_dot(const xm_tensor_t *a, const xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, nblocks;
	xm_scalar_type_t scalartype;
	xm_scalar_t dot = 0;
	size_t i, maxblkbytes, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(b))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(b))
		fatal("tensors must have same scalar type");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	scalartype = xm_tensor_get_scalar_type(a);
	zero = xm_dim_zero(0);
	maxblkbytes = xm_tensor_get_largest_block_bytes(a);
	nblocks = xm_tensor_get_nblocks(a);
	nblklist = xm_dim_dot(&nblocks);
#ifdef _OPENMP
#pragma omp parallel private(i) reduction(+:dot)
#endif
{
	xm_dim_t ia, ib;
	void *buf1, *buf2, *buf3;
	size_t blksize;
	xm_block_type_t blocktype;

	if ((buf1 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	if ((buf2 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	if ((buf3 = malloc(maxblkbytes)) == NULL)
		fatal("out of memory");
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank) {
			xm_scalar_t scalara, scalarb;
			ia = xm_dim_from_offset(i, &nblocks);
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			blocktype = xm_tensor_get_block_type(a, ia);
			if (blocktype == XM_BLOCK_TYPE_ZERO)
				continue;
			blocktype = xm_tensor_get_block_type(b, ib);
			if (blocktype == XM_BLOCK_TYPE_ZERO)
				continue;
			blksize = xm_tensor_get_block_size(b, ib);
			xm_tensor_read_block(b, ib, buf1);
			xm_tensor_unfold_block(b, ib, cidxb, zero, buf1,
			    buf2, blksize);
			xm_tensor_read_block(a, ia, buf1);
			xm_tensor_unfold_block(a, ia, cidxa, zero, buf1,
			    buf3, blksize);
			scalara = xm_tensor_get_block_scalar(a, ia);
			scalarb = xm_tensor_get_block_scalar(b, ib);
			scalara = xm_scalar_mul(scalara, scalarb, scalartype);
			scalarb = xm_scalar_dot(buf2, buf3, blksize,
			    scalartype);
			scalara = xm_scalar_mul(scalara, scalarb, scalartype);
			dot = xm_scalar_add(dot, scalara, scalartype);
		}
	}
	free(buf1);
	free(buf2);
	free(buf3);
}
#ifdef XM_USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &dot, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
	    MPI_COMM_WORLD);
#endif
	return dot;
}

void
xm_print_banner(void)
{
	printf("Libxm Tensor Library\n");
	printf("Copyright (c) 2014-2018 Ilya Kaliman\n");
	printf("https://github.com/ilyak/libxm\n");
	printf("Reference: https://dx.doi.org/10.1002/jcc.24713\n");
}
