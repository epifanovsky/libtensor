#ifndef __LIBTENSOR_TENSOR_OPERATION_I_H
#define __LIBTENSOR_TENSOR_OPERATION_I_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

template<typename T> class tensor_i;

/**	\brief Interface for %tensor operations
	\param T Tensor element type.

	\ingroup libtensor
**/
template<typename T>
class tensor_operation_i {
public:
	typedef T element_t; //!< Tensor element type

	/**	\brief Performs the operation
	**/
	virtual void perform(tensor_i<element_t> &t) throw(exception) = 0;
};

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_OPERATION_I_H

