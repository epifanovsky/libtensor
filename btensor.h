#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_info_i.h"
#include "block_tensor_i.h"
#include "tensor.h"

namespace libtensor {

/**	\brief Block %tensor

	<b>Request to lower symmetry (req_lower_symmetry)</b>

	Lowers the permutational symmetry of the block tensor to the requested
	or lower, if necessary.

	\ingroup libtensor
**/
template<size_t N, typename T, typename Alloc>
class btensor : public btensor_i<N,T> {
private:
	tensor<N,T,Alloc> m_t; //<! Underlying tensor for stub implementation

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	**/
	btensor(const block_info_i<N> &bi);

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	**/
	btensor(const btensor_i<N,T> &bt);

	/**	\brief Stub constructor, to be removed later
	**/
	btensor(const dimensions<N> &d);

	/**	\brief Virtual destructor
	**/
	virtual ~btensor();
	//@}

	//!	\name Implementation of tensor_i<T>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

protected:
	//!	\name Implementation of tensor_i<T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *ptr) throw(exception);
	//@}

	//!	\name Implementation of block_tensor_i<T>
	//@{
	virtual void on_req_symmetry(const symmetry_i<N> &sym) throw(exception);
	virtual tensor_i<N,T> &on_req_unique_block(const index<N> &idx)
		throw(exception);
	//@}
};

template<size_t N, typename T, typename Alloc>
inline btensor<N,T,Alloc>::btensor(const dimensions<N> &d) : m_t(d) {
}

template<size_t N, typename T, typename Alloc>
btensor<N,T,Alloc>::~btensor() {
}

template<size_t N, typename T, typename Alloc>
const dimensions<N> &btensor<N,T,Alloc>::get_dims() const {
	return m_t.get_dims();
}

template<size_t N, typename T, typename Alloc>
void btensor<N,T,Alloc>::on_req_prefetch() throw(exception) {
	throw_exc("block_tensor<N,T,Alloc>", "on_req_prefetch()",
		"Unhandled event");
}

template<size_t N, typename T, typename Alloc>
T *btensor<N,T,Alloc>::on_req_dataptr() throw(exception) {
	throw_exc("block_tensor<N,T,Alloc>", "on_req_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Alloc>
const T *btensor<N,T,Alloc>::on_req_const_dataptr() throw(exception) {
	throw_exc("block_tensor<N,T,Alloc>", "on_req_const_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Alloc>
void btensor<N,T,Alloc>::on_ret_dataptr(const T *ptr) throw(exception) {
	throw_exc("block_tensor<N,T,Alloc>", "on_ret_dataptr(const T*)",
		"Unhandled event");
}

template<size_t N, typename T, typename Alloc>
void btensor<N,T,Alloc>::on_req_symmetry(const symmetry_i<N> &sym)
	throw(exception) {
	throw_exc("block_tensor<N,T,Alloc>",
		"on_req_symmetry(const symmetry_i<N>&)",
		"Unhandled event");
}

template<size_t N, typename T, typename Alloc>
tensor_i<N,T> &btensor<N,T,Alloc>::on_req_unique_block(const index<N> &idx)
	throw(exception) {
	if(m_t.get_dims().abs_index(idx) != 0) {
		throw_exc("block_tensor<N,T,Alloc>",
			"on_req_unique_block(const index<N>&)",
			"Stub implementation only returns the zeroth block");
	}
	return m_t;
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

