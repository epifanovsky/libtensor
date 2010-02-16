#ifndef LIBTENSOR_BLAS_MKL_H
#define LIBTENSOR_BLAS_MKL_H

#include <mkl.h>

namespace libtensor {


/**	\brief BLAS function dcopy (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dcopy(
	size_t n, const double *dx, size_t incx, double *dy, size_t incy) {

	MKL_INT mkl_n = n;
	MKL_INT mkl_incx = incx;
	MKL_INT mkl_incy = incy;
	cblas_dcopy(mkl_n, dx, mkl_incx, dy, mkl_incy);
}


/**	\brief BLAS function dscal (Intel MKL)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {

	MKL_INT mkl_n = n;
	MKL_INT mkl_incx = incx;
	cblas_dscal(mkl_n, da, dx, mkl_incx);
}


} // namespace libtensor

#endif // LIBTENSOR_BLAS_MKL_H
