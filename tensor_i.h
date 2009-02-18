#ifndef __LIBTENSOR_TENSOR_I_H
#define __LIBTENSOR_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"

namespace libtensor {

template<typename T> class tensor_operation_i;
template<typename T> class tensor_operation_handler_i;
template<typename T> class tensor_operation_dispatcher;

/**	\brief Abstract %tensor interface
	\param T Tensor element type
**/
template<typename T>
class tensor_i {
	friend tensor_operation_dispatcher<T>;

public:
	//!	Tensor operation type
	typedef tensor_operation_i<T> tensor_operation_t;

public:
	/**	\brief Returns the %dimensions of the %tensor
	**/
	virtual const dimensions &get_dims() const = 0;

	/**	\brief Performs an arbitrary operation on the %tensor
		\param op Tensor operation
	**/
	virtual void operation(tensor_operation_t &op) throw(exception) = 0;

protected:
	/**	\brief Returns the %tensor's operation handler
	**/
	virtual const tensor_operation_handler_i<T>
		&get_tensor_operation_handler() const = 0;
};

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_I_H

