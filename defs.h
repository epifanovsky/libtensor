#ifndef LIBTENSOR_DEFS_H
#define LIBTENSOR_DEFS_H

#include <cstddef>

/**	\brief Tensor library
	\ingroup libtensor
**/
namespace libtensor {

/**	\brief Limits the maximum order of tensors
**/
const size_t max_tensor_order = 6;

}

#undef TENSOR_DEBUG
#undef LIBTENSOR_DEBUG
#ifdef DEBUG_CHECK
#define TENSOR_DEBUG
#define LIBTENSOR_DEBUG
#endif

#ifdef USE_MKL
#include <mkl.h>
#undef USE_BLAS
#endif
#ifdef USE_BLAS
#include <cblas.h>
#endif

/**	\defgroup libtensor Tensor library
**/

#endif // LIBTENSOR_DEFS_H

