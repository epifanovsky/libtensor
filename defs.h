#ifndef __TENSOR_DEFS_H
#define __TENSOR_DEFS_H

#include <cstddef>

/**	\brief Tensor library
	\ingroup tensor
**/
namespace libtensor {

const unsigned int max_tensor_order = 6; //!< Limits the maximum order of tensors

/**	\brief Default integral type used for %tensor %index classes
	\ingroup tensor

	This integral type is used by default for %tensor %index elements and
	therefore limits the size of tensors along each dimension.
**/
typedef unsigned long default_index_t;

}

#undef TENSOR_DEBUG
#ifdef DEBUG_CHECK
#define TENSOR_DEBUG
#endif

/**	\defgroup tensor Tensor library
**/

/**	\defgroup tensor_interfaces Interfaces of the tensor library
	\ingroup tensor
**/

/**	\defgroup tensor_tod Tensor operations (tensor element type: double)
	\ingroup tensor
**/

/**	\defgroup tensor_vmm Virtual memory manager
	\ingroup tensor

**/

#endif // __TENSOR_DEFS_H

