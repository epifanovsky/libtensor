#ifndef LIBTENSOR_DEFS_H
#define LIBTENSOR_DEFS_H

#include <cstddef>

/**	\brief Tensor library
	\ingroup libtensor
**/
namespace libtensor {

/**	\brief Limits the maximum order of tensors
**/
const unsigned int max_tensor_order = 6;

}

#undef TENSOR_DEBUG
#ifdef DEBUG_CHECK
#define TENSOR_DEBUG
#endif

/**	\defgroup libtensor Tensor library
**/

#endif // LIBTENSOR_DEFS_H

