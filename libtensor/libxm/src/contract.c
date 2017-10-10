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

#include <stdlib.h>
#include <string.h>

#ifdef XM_USE_MPI
#include <mpi.h>
#endif

#include "xm.h"
#include "util.h"

struct blockpair {
	xm_dim_t blkidxa, blkidxb;
	xm_scalar_t alpha;
};

void sgemm_(char *, char *, long int *, long int *, long int *,
    float *, float *, long int *,
    float *, long int *, float *,
    float *, long int *);
void cgemm_(char *, char *, long int *, long int *, long int *,
    float complex *, float complex *, long int *,
    float complex *, long int *, float complex *,
    float complex *, long int *);
void dgemm_(char *, char *, long int *, long int *, long int *,
    double *, double *, long int *,
    double *, long int *, double *,
    double *, long int *);
void zgemm_(char *, char *, long int *, long int *, long int *,
    double complex *, double complex *, long int *,
    double complex *, long int *, double complex *,
    double complex *, long int *);

static void
xgemm(char transa, char transb, long int m, long int n, long int k,
    xm_scalar_t alpha, void *a, long int lda, void *b, long int ldb,
    xm_scalar_t beta, void *c, long int ldc, int type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float al = alpha, bt = beta;
		sgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex al = alpha, bt = beta;
		cgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double al = alpha, bt = beta;
		dgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex al = alpha, bt = beta;
		zgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	}
}

static void
compute_block(xm_scalar_t alpha, const xm_tensor_t *a, const xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, xm_dim_t cidxa, xm_dim_t aidxa,
    xm_dim_t cidxb, xm_dim_t aidxb, xm_dim_t cidxc, xm_dim_t aidxc,
    xm_dim_t blkidxc, struct blockpair *pairs, void *buf)
{
	size_t maxblockbytesa = xm_tensor_get_largest_block_bytes(a);
	size_t maxblockbytesb = xm_tensor_get_largest_block_bytes(b);
	size_t maxblockbytesc = xm_tensor_get_largest_block_bytes(c);
	xm_dim_t dims, blkidxa, blkidxb, nblocksa, nblocksb;
	void *bufa1, *bufa2, *bufb1, *bufb2, *bufc1, *bufc2;
	size_t i, j, m, n, k, nblkk, blksize;
	int type;

	bufa1 = buf;
	bufa2 = (char *)bufa1 + maxblockbytesa;
	bufb1 = (char *)bufa2 + maxblockbytesa;
	bufb2 = (char *)bufb1 + maxblockbytesb;
	bufc1 = (char *)bufb2 + maxblockbytesb;
	bufc2 = (char *)bufc1 + maxblockbytesc;

	type = xm_tensor_get_scalar_type(c);
	nblocksa = xm_tensor_get_nblocks(a);
	nblocksb = xm_tensor_get_nblocks(b);
	nblkk = xm_dim_dot_mask(&nblocksa, &cidxa);

	dims = xm_tensor_get_block_dims(c, blkidxc);
	m = xm_dim_dot_mask(&dims, &cidxc);
	n = xm_dim_dot_mask(&dims, &aidxc);
	xm_tensor_read_block(c, blkidxc, bufc2);
	if (aidxc.n > 0 && aidxc.i[0] == 0)
		xm_tensor_unfold_block(c, blkidxc, aidxc, cidxc,
		    bufc2, bufc1, n);
	else
		xm_tensor_unfold_block(c, blkidxc, cidxc, aidxc,
		    bufc2, bufc1, m);
	blksize = xm_tensor_get_block_size(c, blkidxc);
	xm_scalar_mul(bufc1, blksize, type, beta);
	if (alpha == 0)
		goto done;
	blkidxa = xm_dim_zero(nblocksa.n);
	blkidxb = xm_dim_zero(nblocksb.n);
	xm_dim_set_mask(&blkidxa, &aidxa, &blkidxc, &cidxc);
	xm_dim_set_mask(&blkidxb, &aidxb, &blkidxc, &aidxc);
	for (i = 0; i < nblkk; i++) {
		int blktypea = xm_tensor_get_block_type(a, blkidxa);
		int blktypeb = xm_tensor_get_block_type(b, blkidxb);
		pairs[i].alpha = 0;
		pairs[i].blkidxa = blkidxa;
		pairs[i].blkidxb = blkidxb;
		if (blktypea != XM_BLOCK_TYPE_ZERO &&
		    blktypeb != XM_BLOCK_TYPE_ZERO) {
			xm_scalar_t sa = xm_tensor_get_block_scalar(a, blkidxa);
			xm_scalar_t sb = xm_tensor_get_block_scalar(b, blkidxb);
			pairs[i].alpha = sa * sb;
		}
		xm_dim_inc_mask(&blkidxa, &nblocksa, &cidxa);
		xm_dim_inc_mask(&blkidxb, &nblocksb, &cidxb);
	}
	for (i = 0; i < nblkk; i++) {
		if (pairs[i].alpha == 0)
			continue;
		for (j = i+1; j < nblkk; j++) {
			xm_dim_t dia, dja, dib, djb, pia, pja, pib, pjb;
			size_t ii, good = 1;
			if (pairs[j].alpha == 0)
				continue;
			dia = pairs[i].blkidxa;
			dja = pairs[j].blkidxa;
			dib = pairs[i].blkidxb;
			djb = pairs[j].blkidxb;
			if (xm_tensor_get_block_data_ptr(a, dia) !=
			    xm_tensor_get_block_data_ptr(a, dja) ||
			    xm_tensor_get_block_data_ptr(b, dib) !=
			    xm_tensor_get_block_data_ptr(b, djb))
				continue;
			pia = xm_tensor_get_block_permutation(a, dia);
			pja = xm_tensor_get_block_permutation(a, dja);
			pib = xm_tensor_get_block_permutation(b, dib);
			pjb = xm_tensor_get_block_permutation(b, djb);
			for (ii = 0; ii < aidxa.n && good; ii++) {
				if (pia.i[aidxa.i[ii]] != pja.i[aidxa.i[ii]])
					good = 0;
			}
			for (ii = 0; ii < aidxb.n && good; ii++) {
				if (pib.i[aidxb.i[ii]] != pjb.i[aidxb.i[ii]])
					good = 0;
			}
			if (good) {
				pairs[i].alpha += pairs[j].alpha;
				pairs[j].alpha = 0;
			}
		}
	}
	for (i = 0; i < nblkk; i++) {
		if (pairs[i].alpha != 0) {
			blkidxa = pairs[i].blkidxa;
			blkidxb = pairs[i].blkidxb;
			dims = xm_tensor_get_block_dims(a, blkidxa);
			k = xm_dim_dot_mask(&dims, &cidxa);

			xm_tensor_read_block(a, blkidxa, bufa1);
			xm_tensor_unfold_block(a, blkidxa, cidxa,
			    aidxa, bufa1, bufa2, k);
			xm_tensor_read_block(b, blkidxb, bufb1);
			xm_tensor_unfold_block(b, blkidxb, cidxb,
			    aidxb, bufb1, bufb2, k);

			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				xgemm('T', 'N', (int)n, (int)m, (int)k,
				    alpha*pairs[i].alpha, bufb2, (int)k, bufa2,
				    (int)k, 1.0, bufc1, (int)n, type);
			} else {
				xgemm('T', 'N', (int)m, (int)n, (int)k,
				    alpha*pairs[i].alpha, bufa2, (int)k, bufb2,
				    (int)k, 1.0, bufc1, (int)m, type);
			}
		}
	}
done:
	if (aidxc.n > 0 && aidxc.i[0] == 0)
		xm_tensor_fold_block(c, blkidxc, aidxc, cidxc, bufc1, bufc2, n);
	else
		xm_tensor_fold_block(c, blkidxc, cidxc, aidxc, bufc1, bufc2, m);
	xm_tensor_write_block(c, blkidxc, bufc2);
}

void
xm_contract(xm_scalar_t alpha, const xm_tensor_t *a, const xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	const xm_block_space_t *bsa, *bsb, *bsc;
	xm_dim_t nblocksa, cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *blklist;
	size_t i, bufbytes, nblkk, nblklist;
	int mpirank = 0, mpisize = 1;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(c) ||
	    xm_tensor_get_allocator(b) != xm_tensor_get_allocator(c))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(c) ||
	    xm_tensor_get_scalar_type(b) != xm_tensor_get_scalar_type(c))
		fatal("tensors must have same scalar type");
#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	bsc = xm_tensor_get_block_space(c);

	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("bad contraction indices");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("bad contraction indices");
	if (strlen(idxc) != xm_block_space_get_ndims(bsc))
		fatal("bad contraction indices");

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	xm_make_masks(idxc, idxa, &cidxc, &aidxa);
	xm_make_masks(idxc, idxb, &aidxc, &aidxb);

	if (aidxa.n + cidxa.n != xm_block_space_get_ndims(bsa))
		fatal("bad contraction indices");
	if (aidxb.n + cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("bad contraction indices");
	if (aidxc.n + cidxc.n != xm_block_space_get_ndims(bsc))
		fatal("bad contraction indices");
	if (!(aidxc.n > 0 && aidxc.i[0] == 0) &&
	    !(cidxc.n > 0 && cidxc.i[0] == 0))
		fatal("bad contraction indices");

	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent a and b tensor block-spaces");
	for (i = 0; i < cidxc.n; i++)
		if (!xm_block_space_eq1(bsc, cidxc.i[i], bsa, aidxa.i[i]))
			fatal("inconsistent a and c tensor block-spaces");
	for (i = 0; i < aidxc.n; i++)
		if (!xm_block_space_eq1(bsc, aidxc.i[i], bsb, aidxb.i[i]))
			fatal("inconsistent b and c tensor block-spaces");

	nblocksa = xm_tensor_get_nblocks(a);
	nblkk = xm_dim_dot_mask(&nblocksa, &cidxa);
	bufbytes = 2 * (xm_tensor_get_largest_block_bytes(a) +
			xm_tensor_get_largest_block_bytes(b) +
			xm_tensor_get_largest_block_bytes(c));
	xm_tensor_get_canonical_block_list(c, &blklist, &nblklist);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	struct blockpair *pairs;
	void *buf;

	if ((pairs = malloc(nblkk * sizeof *pairs)) == NULL)
		fatal("out of memory");
	if ((buf = malloc(bufbytes)) == NULL)
		fatal("out of memory");
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nblklist; i++) {
		if ((int)i % mpisize == mpirank)
			compute_block(alpha, a, b, beta, c, cidxa, aidxa, cidxb,
			    aidxb, cidxc, aidxc, blklist[i], pairs, buf);
	}
	free(buf);
	free(pairs);
}
	free(blklist);
#ifdef XM_USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}
