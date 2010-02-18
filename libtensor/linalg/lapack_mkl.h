#ifndef LIBTENSOR_LAPACK_MKL_H
#define LIBTENSOR_LAPACK_MKL_H

#include <mkl.h>

namespace libtensor {


/**	\brief LAPACK function dgesv (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesv(size_t n, size_t nrhs, double *a, size_t lda,
	int *ipiv, double *b, size_t ldb) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_nrhs = nrhs;
//	MKL_INT mkl_lda = lda;
//	MKL_INT mkl_ldb = ldb;
//	MKL_INT mkl_info = 0;
	int mkl_n = n;
	int mkl_nrhs = nrhs;
	int mkl_lda = lda;
	int mkl_ldb = ldb;
	int mkl_info = 0;
	dgesv_(&mkl_n, &mkl_nrhs, a, &mkl_lda, ipiv, b, &mkl_ldb, &mkl_info);
	return mkl_info;
}


/**	\brief LAPACK function dgesvd (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
	size_t lda, double *s, double *u, size_t ldu, double *vt,
	size_t ldvt, double *work, size_t lwork) {

//	MKL_INT mkl_m = m;
//	MKL_INT mkl_n = n;
//	MKL_INT mkl_lda = lda;
//	MKL_INT mkl_ldu = ldu;
//	MKL_INT mkl_ldvt = ldvt;
//	MKL_INT mkl_lwork = lwork;
//	MKL_INT mkl_info = 0;
	int mkl_m = m;
	int mkl_n = n;
	int mkl_lda = lda;
	int mkl_ldu = ldu;
	int mkl_ldvt = ldvt;
	int mkl_lwork = lwork;
	int mkl_info = 0;
	dgesvd_(&jobu, &jobvt, &mkl_m, &mkl_n, a, &mkl_lda, s, u, &mkl_ldu,
		vt, &mkl_ldvt, work, &mkl_lwork, &mkl_info);
	return mkl_info;
}


/**	\brief LAPACK function dsyev (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
	double *w, double *work, size_t lwork) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_lda = lda;
//	MKL_INT mkl_lwork = lwork;
//	MKL_INT mkl_info = 0;
	int mkl_n = n;
	int mkl_lda = lda;
	int mkl_lwork = lwork;
	int mkl_info = 0;
	dsyev_(&jobz, &uplo, &mkl_n, a, &mkl_lda, w, work, &mkl_lwork,
		&mkl_info);
	return mkl_info;
}


/**	\brief LAPACK function dgeev (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a, size_t lda,
	double *wr, double *wi, double *vl, size_t ldvl, double *vr,
	size_t ldvr, double *work, size_t lwork) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_lda = lda;
//	MKL_INT mkl_ldvl = ldvl;
//	MKL_INT mkl_ldvr = ldvr;
//	MKL_INT mkl_lwork = lwork;
//	MKL_INT mkl_info = 0;
	int mkl_n = n;
	int mkl_lda = lda;
	int mkl_ldvl = ldvl;
	int mkl_ldvr = ldvr;
	int mkl_lwork = lwork;
	int mkl_info = 0;
	dgeev_(&jobvl, &jobvr, &mkl_n, a, &mkl_lda, wr, wi, vl, &mkl_ldvl,
		vr, &mkl_ldvr, work, &mkl_lwork, &mkl_info);
	return mkl_info;
}


} // namespace libtensor

#endif // LIBTENSOR_LAPACK_MKL_H
