#ifndef LIBTENSOR_TENSOR_I_H
#define LIBTENSOR_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "tensor_operation_handler.h"

namespace libtensor {

template<typename T> class tensor_operation_dispatcher;

/**	\brief Tensor interface
	\param T Tensor element type

	<b>Implementing this interface</b>

	Implementations of this interface must realize two methods:
	get_dims() and get_tensor_operation_handler().

	get_dims() simply returns the %dimensions of the %tensor represented
	by an object.

	get_tensor_operation_handler() returns an instance of
	libtensor::tensor_operation_handler that is responsible for reacting
	to events that arise while a %tensor operation is performing.

	\ingroup libtensor
**/
template<typename T>
class tensor_i {
	friend class tensor_operation_dispatcher<T>;

public:
	/**	\brief Returns the %dimensions of the %tensor
	**/
	virtual const dimensions &get_dims() const = 0;

protected:
	/**	\brief Returns the %tensor's operation handler
	**/
	virtual tensor_operation_handler<T> &get_tensor_operation_handler() = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_I_H

