#ifndef LIBTENSOR_DIRECT_TENSOR_OPERATION_H
#define LIBTENSOR_DIRECT_TENSOR_OPERATION_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Underlying operation for direct tensors

	\param N Tensor order.
	\param T Tensor element type.

	Generally speaking, a %tensor operation can have any number of
	objects as parameters and produce multiple results.
	Directly calculated tensors (implementations of
	libtensor::direct_tensor) can have multiple parameters, but the
	result is a single %tensor.

	Classes that implement underlying operations for direct tensors
	must implement the methods declared here.

	\ingroup libtensor_core
**/
template<size_t N, typename T>
class direct_tensor_operation {
public:
	/**	\brief Invoked to indicate that the operation is to be
			executed soon

		The implementation should pass this event to the parameters
		so they can be pre-fetched if stored on slow media
	**/
	virtual void prefetch() throw(exception) = 0;

	/**	\brief Invoked to execute the operation
		\param t The output %tensor
	**/
	virtual void perform(tensor_i<N,T> &t) throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_TENSOR_OPERATION_H

