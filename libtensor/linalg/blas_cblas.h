#ifndef LIBTENSOR_BLAS_CBLAS_H
#define LIBTENSOR_BLAS_CBLAS_H

#include <cblas.h>

namespace libtensor {

/**	\brief BLAS function dcopy (CBLAS)

	\ingroup libtensor_linalg
 **/
inline void blas_dcopy(
	size_t n, const double *dx, size_t incx, double *dy, size_t incy) {
	cblas_dcopy(n, dx, incx, dy, incy);
}

/**	\brief BLAS function dscal (CBLAS)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {
	cblas_dscal(n, da, dx, incx);
}


/**	\brief BLAS function ddot (CBLAS)

	\ingroup libtensor_linalg
 **/
inline double blas_ddot(size_t n, const double *dx, size_t incx,
	const double *dy, size_t incy) {

	return cblas_ddot(n, dx, incx, dy, incy);
}


/**	\brief BLAS function daxpy (CBLAS)

	\ingroup libtensor_linalg
 **/
inline void blas_daxpy(size_t n, double da, const double *dx, size_t incx,
	double *dy, size_t incy) {

	cblas_daxpy(n, da, dx, incx, dy, incy);
}


/**	\brief BLAS function dgemv (CBLAS)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemv(bool trans, size_t m, size_t n, double alpha,
	const double *da, size_t lda, const double *dx, size_t incx,
	double beta, double *dy, size_t incy) {

	cblas_dgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n,
		alpha, da, lda, dx, incx, beta, dy, incy);
}


/**	\brief BLAS function dgemm (CBLAS)

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

#endif // LIBTENSOR_BLAS_CBLAS_H
