#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include <libvmm.h>
#include "defs.h"
#include "exception.h"
#include "block_info_i.h"
#include "btensor_i.h"
#include "tensor.h"
#include "labeled_btensor.h"

namespace libtensor {

struct btensor_default_traits {
	typedef double element_t;
	typedef libvmm::std_allocator<double> allocator_t;
};

/**	\brief Block %tensor

	<b>Request to lower symmetry (req_lower_symmetry)</b>

	Lowers the permutational symmetry of the block tensor to the requested
	or lower, if necessary.

	\ingroup libtensor
 **/
template<size_t N, typename Traits = btensor_default_traits>
	class btensor : public btensor_i<N, typename Traits::element_t> {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	//! Underlying tensor for stub implementation
	tensor<N, element_t, allocator_t> m_t;

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
	btensor(const btensor_i<N, element_t> &bt);

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

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	template<typename ExprT>
	labeled_btensor<N, Traits, letter_expr<N, ExprT> > operator()(
		letter_expr<N, ExprT> expr) {
		return labeled_btensor<N, Traits, letter_expr<N, ExprT> >(
			*this, expr);
	}

protected:
	//!	\name Implementation of tensor_i<N,T>
	//@{
	virtual void on_req_prefetch() throw (exception);
	virtual element_t *on_req_dataptr() throw (exception);
	virtual const element_t *on_req_const_dataptr() throw (exception);
	virtual void on_ret_dataptr(const element_t *ptr) throw (exception);
	//@}

	//!	\name Implementation of btensor_i<N,T>
	//@{
	virtual void on_req_symmetry(const symmetry_i<N> &sym) throw (exception);
	virtual tensor_i<N, element_t> &on_req_unique_block(const index<N> &idx)
	throw (exception);
	//@}
};

template<size_t N, typename Traits>
inline btensor<N, Traits>::btensor(const dimensions<N> &d) : m_t(d) {
}

template<size_t N, typename Traits>
btensor<N, Traits>::~btensor() {
}

template<size_t N, typename Traits>
const dimensions<N> &btensor<N, Traits>::get_dims() const {
	return m_t.get_dims();
}

template<size_t N, typename Traits> template<typename ExprT>
inline labeled_btensor<N, Traits, letter_expr<N, ExprT> >
btensor<N, Traits>::operator()(letter_expr<N, ExprT> expr) {
	return labeled_btensor<N, Traits, letter_expr<N, ExprT> >(
		*this, expr);
}

template<size_t N, typename Traits>
void btensor<N, Traits>::on_req_prefetch() throw (exception) {
	throw_exc("btensor<N,Traits>", "on_req_prefetch()",
		"Unhandled event");
}

template<size_t N, typename Traits>
typename Traits::element_t *btensor<N, Traits>::on_req_dataptr()
throw (exception) {
	throw_exc("btensor<N,Traits>", "on_req_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename Traits>
const typename Traits::element_t *btensor<N, Traits>::on_req_const_dataptr()
throw (exception) {
	throw_exc("btensor<N,Traits>", "on_req_const_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename Traits>
void btensor<N, Traits>::on_ret_dataptr(const element_t *ptr)
throw (exception) {
	throw_exc("btensor<N,Traits>", "on_ret_dataptr(const T*)",
		"Unhandled event");
}

template<size_t N, typename Traits>
void btensor<N, Traits>::on_req_symmetry(const symmetry_i<N> &sym)
throw (exception) {
	throw_exc("btensor<N,Traits>",
		"on_req_symmetry(const symmetry_i<N>&)",
		"Unhandled event");
}

template<size_t N, typename Traits>
tensor_i<N, typename Traits::element_t>
&btensor<N, Traits>::on_req_unique_block(const index<N> &idx)
throw (exception) {

	if(m_t.get_dims().abs_index(idx) != 0) {
		throw_exc("btensor<N,Traits>",
			"on_req_unique_block(const index<N>&)",
			"Stub implementation only returns the zeroth block");
	}
	return m_t;
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

