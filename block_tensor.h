#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_index_space_i.h"
#include "block_tensor_i.h"
#include "immutable.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Block tensor implementation
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator.

	\ingroup libtensor
 **/
template<size_t N, typename T, typename Alloc>
class block_tensor : public block_tensor_i<N, T>, public immutable {
public:
	tensor<N, T, Alloc> m_t;
	tensor_ctrl<N, T> m_ctrl;

public:
	//!	\name Construction and destruction
	//@{
	block_tensor(const block_index_space_i<N> &bis);
	block_tensor(const block_tensor_i<N, T> &bt);
	block_tensor(const block_tensor<N, T, Alloc> &bt);
	virtual ~block_tensor();
	//@}

	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);
	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable();
	//@}
};

template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_index_space_i<N> &bis) :
	m_t(bis.get_dims()), m_ctrl(m_t) {
}

template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_tensor_i<N, T> &bt) :
	m_t(bt.get_dims()), m_ctrl(m_t) {
}

template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_tensor<N, T, Alloc> &bt) :
	m_t(bt.get_dims()), m_ctrl(m_t) {
}

template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::~block_tensor() {
}

template<size_t N, typename T, typename Alloc>
const dimensions<N> &block_tensor<N, T, Alloc>::get_dims() const {
	return m_t.get_dims();
}

template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &block_tensor<N, T, Alloc>::on_req_block(const index<N> &idx)
	throw(exception) {
	return m_t;
}

template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_prefetch() throw(exception) {
	m_ctrl.req_prefetch();
}

template<size_t N, typename T, typename Alloc>
T *block_tensor<N, T, Alloc>::on_req_dataptr() throw(exception) {
	return m_ctrl.req_dataptr();
}

template<size_t N, typename T, typename Alloc>
const T *block_tensor<N, T, Alloc>::on_req_const_dataptr() throw(exception) {
	return m_ctrl.req_const_dataptr();
}

template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_ret_dataptr(const T *p) throw(exception) {
	m_ctrl.ret_dataptr(p);
}

template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_set_immutable() {
	m_t.set_immutable();
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H
