#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include <libvmm.h>
#include "defs.h"
#include "exception.h"
#include "core/block_index_space.h"
#include "core/block_tensor.h"
#include "core/block_tensor_ctrl.h"
#include "core/immutable.h"
#include "bispace_i.h"
#include "btensor_i.h"
#include "labeled_btensor.h"

namespace libtensor {

template<typename T>
struct btensor_traits {
	typedef T element_t;
	typedef libvmm::std_allocator<T> allocator_t;
};

/**	\brief User-friendly block %tensor

	\ingroup libtensor_iface
 **/
template<size_t N, typename T = double, typename Traits = btensor_traits<T> >
	class btensor : public btensor_i<N, T>, public immutable {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	block_tensor<N, element_t, allocator_t> m_bt;
	tensor_ctrl<N, element_t> m_tctrl;

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	 **/
	btensor(const bispace_i<N> &bi);

	/**	\brief Constructs a block %tensor using a block %index space
		\param bis Block %index space
	 **/
	btensor(const block_index_space<N> &bis);

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	 **/
	btensor(const btensor_i<N, element_t> &bt);

	/**	\brief Virtual destructor
	 **/
	virtual ~btensor();
	//@}

	//!	\name Implementation of tensor_i<N, T>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

	//!	\name Implementation of block_tensor_i<N, T>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	//@}

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	template<typename ExprT>
	labeled_btensor<N, T, true, letter_expr<N, ExprT> > operator()(
		letter_expr<N, ExprT> expr);

protected:
	//!	\name Implementation of libtensor::tensor_i<N,T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *ptr) throw(exception);
	//@}

	//!	\name Implementation of libtensor::block_tensor_i<N,T>
	//@{
	virtual const symmetry_i<N, T> &on_req_symmetry() throw(exception);
	virtual void on_req_symmetry_operation(symmetry_operation_i<N, T> &op)
		throw(exception);
	virtual orbit_iterator<N, T> on_req_orbits() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable();
	//@}
};

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const bispace_i<N> &bispace) :
	m_bt(bispace.get_bis()), m_tctrl(m_bt) {
}

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const block_index_space<N> &bis) :
	m_bt(bis), m_tctrl(m_bt) {
}

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const btensor_i<N, element_t> &bt) :
	m_bt(bt), m_tctrl(m_bt) {
}

template<size_t N, typename T, typename Traits>
btensor<N, T, Traits>::~btensor() {
}

template<size_t N, typename T, typename Traits>
const dimensions<N> &btensor<N, T, Traits>::get_dims() const {
	return m_bt.get_dims();
}

template<size_t N, typename T, typename Traits>
inline const block_index_space<N> &btensor<N, T, Traits>::get_bis() const {
	return m_bt.get_bis();
}

template<size_t N, typename T, typename Traits> template<typename ExprT>
inline labeled_btensor<N, T, true, letter_expr<N, ExprT> >
btensor<N, T, Traits>::operator()(letter_expr<N, ExprT> expr) {
	return labeled_btensor<N, T, true, letter_expr<N, ExprT> >(
		*this, expr);
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_prefetch() throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_prefetch();
}

template<size_t N, typename T, typename Traits>
T *btensor<N, T, Traits>::on_req_dataptr() throw(exception) {
	return m_tctrl.req_dataptr();
}

template<size_t N, typename T, typename Traits>
const T *btensor<N, T, Traits>::on_req_const_dataptr() throw(exception) {
	return m_tctrl.req_const_dataptr();
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_ret_dataptr(const T *ptr) throw(exception) {
	m_tctrl.ret_dataptr(ptr);
}

template<size_t N, typename T, typename Traits>
const symmetry_i<N, T> &btensor<N, T, Traits>::on_req_symmetry()
	throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_symmetry()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_symmetry_operation(
	symmetry_operation_i<N, T> &op) throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_symmetry_operation()", "NIY");
}

template<size_t N, typename T, typename Traits>
orbit_iterator<N, T> btensor<N, T, Traits>::on_req_orbits() throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_orbits()", "NIY");
}

template<size_t N, typename T, typename Traits>
tensor_i<N, T> &btensor<N, T, Traits>::on_req_block(const index<N> &idx)
	throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	return ctrl.req_block(idx);
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_ret_block(const index<N> &idx)
	throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_ret_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
inline void btensor<N, T, Traits>::on_set_immutable() {
	m_bt.set_immutable();
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

