#ifndef LIBTENSOR_BLOCK_TENSOR_I_H
#define LIBTENSOR_BLOCK_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "index.h"
#include "tensor_i.h"

namespace libtensor {

template<size_t N, typename T>
class block_tensor_ctrl;

/**	\brief Block %tensor interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class block_tensor_i : public tensor_i<N, T> {
	friend class block_tensor_ctrl<N, T>;

public:
	/**	\brief Returns the block %index space of the block %tensor
	 **/
	virtual const block_index_space<N> &get_bis() const = 0;

protected:
	//!	\name Event handling
	//@{
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception) = 0;
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_I_H
