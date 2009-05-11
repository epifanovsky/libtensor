#ifndef LIBTENSOR_BLOCK_TENSOR_I_H
#define LIBTENSOR_BLOCK_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "tensor_i.h"

namespace libtensor {

template<size_t N, typename T>
class block_tensor_ctrl;

/**	\brief Block tensor interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor
 **/
template<size_t N, typename T>
class block_tensor_i : public tensor_i<N, T> {
	friend class block_tensor_ctrl<N, T>;

protected:
	//!	\name Event handling
	//@{
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception) = 0;
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_I_H
