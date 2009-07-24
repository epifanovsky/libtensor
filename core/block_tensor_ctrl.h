#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "index.h"

namespace libtensor {

/**	\brief Block %tensor control
	\tparam N Block %tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
**/
template<size_t N, typename T>
class block_tensor_ctrl {
private:
	block_tensor_i<N, T> &m_bt; //!< Controlled block %tensor

public:
	//!	\name Construction and destruction
	//@{
	block_tensor_ctrl(block_tensor_i<N, T> &bt);
	~block_tensor_ctrl();
	//@}

	//!	\name Event forwarding
	//@{
	void req_prefetch() throw(exception);
	tensor_i<N, T> &req_block(const index<N> &idx) throw(exception);
	//@}
};

template<size_t N, typename T>
inline block_tensor_ctrl<N, T>::block_tensor_ctrl(block_tensor_i<N, T> &bt) :
	m_bt(bt) {
}

template<size_t N, typename T>
block_tensor_ctrl<N, T>::~block_tensor_ctrl() {
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N,T>::req_prefetch() throw(exception) {
	m_bt.on_req_prefetch();
}

template<size_t N, typename T>
inline tensor_i<N, T> &block_tensor_ctrl<N, T>::req_block(const index<N> &idx)
	throw(exception) {
	return m_bt.on_req_block(idx);
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

