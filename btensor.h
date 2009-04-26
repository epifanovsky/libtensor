#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include <libvmm.h>
#include "defs.h"
#include "exception.h"
#include "bispace_i.h"
#include "btensor_i.h"
#include "tensor.h"
#include "labeled_btensor.h"

namespace libtensor {

template<typename T>
struct btensor_traits {
	typedef T element_t;
	typedef libvmm::std_allocator<T> allocator_t;
};

/**	\brief Block %tensor

	<b>Request to lower symmetry (req_lower_symmetry)</b>

	Lowers the permutational symmetry of the block tensor to the requested
	or lower, if necessary.

	\ingroup libtensor
 **/
template<size_t N, typename T = double, typename Traits = btensor_traits<T> >
	class btensor : public btensor_i<N, T> {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	rc_ptr< bispace_i<N> > m_bispace; //!< Block index space

	//! Underlying tensor for stub implementation
	tensor<N, element_t, allocator_t> m_t;

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	 **/
	btensor(const bispace_i<N> &bi);

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	 **/
	btensor(const btensor_i<N, element_t> &bt);

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
	labeled_btensor<N, T, Traits, letter_expr<N, ExprT> > operator()(
		letter_expr<N, ExprT> expr);

protected:
	//!	\name Implementation of tensor_i<N,T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *ptr) throw(exception);
	//@}

	//!	\name Implementation of btensor_i<N,T>
	//@{
	virtual void on_req_symmetry(const symmetry_i<N> &sym) throw(exception);
	virtual tensor_i<N, T> &on_req_unique_block(const index<N> &idx)
	throw(exception);
	//@}
};

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const bispace_i<N> &bispace) :
m_bispace(bispace.clone()), m_t(bispace.dims()) {
}

template<size_t N, typename T, typename Traits>
btensor<N, T, Traits>::~btensor() {
}

template<size_t N, typename T, typename Traits>
const dimensions<N> &btensor<N, T, Traits>::get_dims() const {
	return m_t.get_dims();
}

template<size_t N, typename T, typename Traits> template<typename ExprT>
inline labeled_btensor<N, T, Traits, letter_expr<N, ExprT> >
btensor<N, T, Traits>::operator()(letter_expr<N, ExprT> expr) {
	return labeled_btensor<N, T, Traits, letter_expr<N, ExprT> >(
		*this, expr);
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_prefetch() throw(exception) {
	throw_exc("btensor<N,T,Traits>", "on_req_prefetch()",
		"Unhandled event");
}

template<size_t N, typename T, typename Traits>
T *btensor<N, T, Traits>::on_req_dataptr() throw(exception) {
	throw_exc("btensor<N,T,Traits>", "on_req_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Traits>
const T *btensor<N, T, Traits>::on_req_const_dataptr() throw(exception) {
	throw_exc("btensor<N,T,Traits>", "on_req_const_dataptr()",
		"Unhandled event");
	return NULL;
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_ret_dataptr(const T *ptr) throw(exception) {
	throw_exc("btensor<N,T,Traits>", "on_ret_dataptr(const T*)",
		"Unhandled event");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_symmetry(const symmetry_i<N> &sym)
throw(exception) {
	throw_exc("btensor<N,T,Traits>",
		"on_req_symmetry(const symmetry_i<N>&)",
		"Unhandled event");
}

template<size_t N, typename T, typename Traits>
tensor_i<N, T> &btensor<N, T, Traits>::on_req_unique_block(const index<N> &idx)
throw(exception) {

	if(m_t.get_dims().abs_index(idx) != 0) {
		throw_exc("btensor<N,T,Traits>",
			"on_req_unique_block(const index<N>&)",
			"Stub implementation only returns the zeroth block");
	}
	return m_t;
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

