#ifndef __LIBTENSOR_TENSOR_I_H
#define __LIBTENSOR_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "tensor_operation_handler.h"

namespace libtensor {

template<typename T> class tensor_operation_dispatcher;

/**	\brief Tensor interface
	\param T Tensor element type

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

#endif // __LIBTENSOR_TENSOR_I_H

