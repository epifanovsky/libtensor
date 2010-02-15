#ifndef LIBTENSOR_BLAS_MKL_H
#define LIBTENSOR_BLAS_MKL_H

#include <mkl.h>

namespace libtensor {


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
