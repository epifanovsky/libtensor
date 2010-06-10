#ifndef LIBTENSOR_BLAS_MKL_H
#define LIBTENSOR_BLAS_MKL_H

#include <mkl.h>

namespace libtensor {


/**	\brief BLAS function dcopy (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dcopy(
	size_t n, const double *dx, size_t incx, double *dy, size_t incy) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_incx = incx;
//	MKL_INT mkl_incy = incy;
	int mkl_n = n;
	int mkl_incx = incx;
	int mkl_incy = incy;
	cblas_dcopy(mkl_n, dx, mkl_incx, dy, mkl_incy);
}


/**	\brief BLAS function dscal (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_incx = incx;
	int mkl_n = n;
	int mkl_incx = incx;
	cblas_dscal(mkl_n, da, dx, mkl_incx);
}


/**	\brief BLAS function ddot (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline double blas_ddot(size_t n, const double *dx, size_t incx, const double *dy,
	size_t incy) {

	return cblas_ddot(n, dx, incx, dy, incy);
}


/**	\brief BLAS function daxpy (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_daxpy(size_t n, double da, const double *dx, size_t incx,
	double *dy, size_t incy) {

//	MKL_INT mkl_n = n;
//	MKL_INT mkl_incx = incx;
//	MKL_INT mkl_incy = incy;
	int mkl_n = n;
	int mkl_incx = incx;
	int mkl_incy = incy;
	cblas_daxpy(mkl_n, da, dx, mkl_incx, dy, mkl_incy);
}


/**	\brief BLAS function dgemv (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemv(bool trans, size_t m, size_t n, double alpha,
	const double *da, size_t lda, const double *dx, size_t incx,
	double beta, double *dy, size_t incy) {

	cblas_dgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n,
		alpha, da, lda, dx, incx, beta, dy, incy);
}


/**	\brief BLAS function dgemm (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemm(bool transa, bool transb, size_t m, size_t n, size_t k,
	double alpha, const double *da, size_t lda, const double *db,
	size_t ldb, double beta, double *dc, size_t ldc) {

	cblas_dgemm(CblasRowMajor, transa ? CblasTrans : CblasNoTrans,
		transb ? CblasTrans : CblasNoTrans, m, n, k, alpha, da, lda,
		db, ldb, beta, dc, ldc);
}


} // namespace libtensor

#endif // LIBTENSOR_BLAS_MKL_H
