#ifndef LIBTENSOR_BLAS_QCHEM_H
#define LIBTENSOR_BLAS_QCHEM_H

#include <qchem.h>
#include <libmathtools/general/blas_include.h>

namespace libtensor {


/**	\brief BLAS function dcopy (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dcopy(size_t n, const double *dx, size_t incx, double *dy,
	size_t incy) {

	CL_DCOPY(n, (double*)dx, incx, dy, incy);
}


/**	\brief BLAS function dscal (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {

	CL_DSCAL(n, da, dx, incx);
}


/**	\brief BLAS function ddot (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline double blas_ddot(size_t n, const double *dx, size_t incx,
	const double *dy, size_t incy) {

	return CL_DDOT(n, (double*)dx, incx, (double*)dy, incy);
}


/**	\brief BLAS function daxpy (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_daxpy(size_t n, double da, const double *dx, size_t incx,
	double *dy, size_t incy) {

	CL_DAXPY(n, da, (double*)dx, incx, dy, incy);
}


/**	\brief BLAS function dgemv (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemv(bool trans, size_t m, size_t n, double alpha,
	const double *da, size_t lda, const double *dx, size_t incx,
	double beta, double *dy, size_t incy) {

	CL_DGEMV(trans ? 'N' : 'T', m, n, alpha, (double*)da, lda, (double*)dx,
		incx, beta, dy, incy);
}


/**	\brief BLAS function dgemm (Q-Chem)

	\ingroup libtensor_linalg
 **/
inline void blas_dgemm(bool transa, bool transb, size_t m, size_t n, size_t k,
	double alpha, const double *da, size_t lda, const double *db,
	size_t ldb, double beta, double *dc, size_t ldc) {

	CL_DGEMM(transa ? 'N' : 'T', transb ? 'N' : 'T', m, n, k, alpha,
		(double*)da, lda, (double*)db, ldb, beta, dc, ldc);
}


} // namespace libtensor

#endif // LIBTENSOR_BLAS_QCHEM_H
