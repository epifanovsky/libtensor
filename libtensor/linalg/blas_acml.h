#ifndef LIBTENSOR_BLAS_ACML_H
#define LIBTENSOR_BLAS_ACML_H

#include <acml.h>

namespace libtensor {


/**	\brief BLAS function dcopy (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dcopy(size_t n, const double *dx, size_t incx, double *dy,
	size_t incy) {

	dcopy(n, (double*)dx, incx, dy, incy);
}


/**	\brief BLAS function dscal (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {

	dscal(n, da, dx, incx);
}


/**	\brief BLAS function ddot (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline double blas_ddot(size_t n, const double *dx, size_t incx,
	const double *dy, size_t incy) {

	return ddot(n, (double*)dx, incx, (double*)dy, incy);
}


/**	\brief BLAS function daxpy (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_daxpy(size_t n, double da, const double *dx, size_t incx,
	double *dy, size_t incy) {

	daxpy(n, da, (double*)dx, incx, dy, incy);
}


/**	\brief BLAS function dgemv (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemv(bool trans, size_t m, size_t n, double alpha,
	const double *da, size_t lda, const double *dx, size_t incx,
	double beta, double *dy, size_t incy) {

	dgemv(trans ? 'N' : 'T', n, m, alpha, (double*)da, lda, (double*)dx,
		incx, beta, dy, incy);
}


/**	\brief BLAS function dgemm (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemm(bool transa, bool transb, size_t m, size_t n, size_t k,
	double alpha, const double *da, size_t lda, const double *db,
	size_t ldb, double beta, double *dc, size_t ldc) {

	dgemm(transb ? 'T' : 'N', transa ? 'T' : 'N', n, m, k, alpha,
		(double*)db, ldb, (double*)da, lda, beta, dc, ldc);
}


} // namespace libtensor

#endif // LIBTENSOR_BLAS_ACML_H
