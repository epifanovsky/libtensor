#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

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
template<typename T, typename Alloc>
class block_tensor : public block_tensor_i<T> {
private:
	tensor<T,Alloc> m_t; //<! Underlying tensor for stub implementation

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	**/
	block_tensor(const block_info_i &bi);

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	**/
	block_tensor(const block_tensor_i<T> &bt);

	/**	\brief Stub constructor, to be removed later
	**/
	block_tensor(const dimensions &d);

	/**	\brief Virtual destructor
	**/
	virtual ~block_tensor();
	//@}

	//!	\name Implementation of tensor_i<T>
	//@{
	virtual const dimensions &get_dims() const;
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
	virtual void on_req_symmetry(const symmetry_i &sym) throw(exception);
	virtual tensor_i<T> &on_req_unique_block(const index &idx)
		throw(exception);
	//@}
};

template<typename T, typename Alloc>
inline block_tensor<T,Alloc>::block_tensor(const dimensions &d) : m_t(d) {
}

template<typename T, typename Alloc>
block_tensor<T,Alloc>::~block_tensor() {
}

template<typename T, typename Alloc>
const dimensions &block_tensor<T,Alloc>::get_dims() const {
	return m_t.get_dims();
}

template<typename T, typename Alloc>
void block_tensor<T,Alloc>::on_req_prefetch() throw(exception) {
	throw_exc("block_tensor<T,Alloc>", "on_req_prefetch()",
		"Unhandled event");
}

template<typename T, typename Alloc>
T *block_tensor<T,Alloc>::on_req_dataptr() throw(exception) {
	throw_exc("block_tensor<T,Alloc>", "on_req_dataptr()",
		"Unhandled event");
}

template<typename T, typename Alloc>
const T *block_tensor<T,Alloc>::on_req_const_dataptr() throw(exception) {
	throw_exc("block_tensor<T,Alloc>", "on_req_const_dataptr()",
		"Unhandled event");
}

template<typename T, typename Alloc>
void block_tensor<T,Alloc>::on_ret_dataptr(const T *ptr) throw(exception) {
	throw_exc("block_tensor<T,Alloc>", "on_ret_dataptr(const T*)",
		"Unhandled event");
}

template<typename T, typename Alloc>
void block_tensor<T,Alloc>::on_req_symmetry(const symmetry_i &sym)
	throw(exception) {
	throw_exc("block_tensor<T,Alloc>", "on_req_symmetry(const symmetry_i&)",
		"Unhandled event");
}

template<typename T, typename Alloc>
tensor_i<T> &block_tensor<T,Alloc>::on_req_unique_block(const index &idx)
	throw(exception) {
	if(m_t.get_dims().abs_index(idx) != 0) {
		throw_exc("block_tensor<T,Alloc>",
			"on_req_unique_block(const index&)",
			"Stub implementation only returns the zeroth block");
	}
	return m_t;
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H

