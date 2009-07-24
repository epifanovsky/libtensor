#ifndef LIBTENSOR_TENSOR_I_H
#define LIBTENSOR_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"

namespace libtensor {

template<size_t N, typename T> class tensor_ctrl;

/**	\brief Tensor interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

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
template<size_t N, typename T>
class tensor_i {
	friend class tensor_ctrl<N, T>;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Virtual destructor
	 **/
	virtual ~tensor_i() { };

	//@}


	//!	\name Tensor dimensions
	//@{

	/**	\brief Returns the %dimensions of the %tensor
	**/
	virtual const dimensions<N> &get_dims() const = 0;

	//@}

protected:
	//!	\name Event handling
	//@{

	/**	\brief Handles the prefetching of %tensor data
	 **/
	virtual void on_req_prefetch() throw(exception) = 0;

	virtual T *on_req_dataptr() throw(exception) = 0;
	virtual const T *on_req_const_dataptr() throw(exception) = 0;
	virtual void on_ret_dataptr(const T *p) throw(exception) = 0;
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_I_H

