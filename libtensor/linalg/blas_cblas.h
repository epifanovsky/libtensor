#ifndef LIBTENSOR_BLAS_CBLAS_H
#define LIBTENSOR_BLAS_CBLAS_H

#include <cblas.h>

namespace libtensor {


/**	\brief BLAS function dscal (CBLAS)

	\ingroup libtensor_linalg
 **/
inline void blas_dscal(size_t n, double da, double *dx, size_t incx) {
	cblas_dscal(n, da, dx, incx);
}


} // namespace libtensor

#endif // LIBTENSOR_BLAS_CBLAS_H
