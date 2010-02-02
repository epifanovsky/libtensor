#ifndef LIBTENSOR_LINALG_H
#define LIBTENSOR_LINALG_H

/**	\defgroup libtensor_linalg Wrappers for linear algebra primitives
	\ingroup libtensor
 **/


#ifdef USE_MKL
#include "linalg/blas_mkl.h"
#include "linalg/lapack_mkl.h"
#else // USE_MKL
#include "linalg/lapack_generic.h"
#endif // USE_MKL

#endif // LIBTENSOR_LINALG_H
