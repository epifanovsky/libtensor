#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "block_tensor_i.h"

namespace libtensor {

/**	\brief Direct block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Op Tensor operation type.
	\tparam Alloc Memory allocator type.

	\ingroup libtensor
**/
template<size_t N, typename T, typename Op, typename Alloc>
class direct_block_tensor : public block_tensor_i<N, T> {
public:
	typedef T element_t; //!< Tensor element type
	typedef Op operation_t; //!< Operation type

private:
	block_index_space<N> m_bis; //!< Block index space
	operation_t m_op; //!< Underlying block tensor operation

public:
	//!	\name Construction and destruction
	//@{
	direct_block_tensor(const block_index_space<N> &bis,
		const operation_t &op);
	virtual ~direct_block_tensor();
	//@}

	//!	\name Implementation of tensor_i<N, T>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

protected:
	//!	\name Implementation of block_tensor_i<N, T>
	//@{
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of tensor_i<N, T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);
	//@}
};

template<size_t N, typename T, typename Op, typename Alloc>
direct_block_tensor<N, T, Op, Alloc>::direct_block_tensor(
	const block_index_space<N> &bis, const operation_t &op) :
	m_bis(bis), m_op(op) {
}

template<size_t N, typename T, typename Op, typename Alloc>
direct_block_tensor<N, T, Op, Alloc>::~direct_block_tensor() {

}

template<size_t N, typename T, typename Op, typename Alloc>
const dimensions<N> &direct_block_tensor<N, T, Op, Alloc>::get_dims() const {
	return m_bis.get_dims();
}

template<size_t N, typename T, typename Op, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Op, Alloc>::on_req_block(
	const index<N> &idx) throw(exception) {

}

template<size_t N, typename T, typename Op, typename Alloc>
void direct_block_tensor<N, T, Op, Alloc>::on_req_prefetch() throw(exception) {

}

template<size_t N, typename T, typename Op, typename Alloc>
T *direct_block_tensor<N, T, Op, Alloc>::on_req_dataptr() throw(exception) {

}

template<size_t N, typename T, typename Op, typename Alloc>
const T *direct_block_tensor<N, T, Op, Alloc>::on_req_const_dataptr()
	throw(exception) {
}

template<size_t N, typename T, typename Op, typename Alloc>
void direct_block_tensor<N, T, Op, Alloc>::on_ret_dataptr(const T *p)
	throw(exception) {
}

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
