#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include <libvmm/libvmm.h>
#include "defs.h"
#include "exception.h"
#include "core/block_index_space.h"
#include "core/block_tensor.h"
#include "core/block_tensor_ctrl.h"
#include "core/immutable.h"
#include "bispace.h"
#include "btensor_i.h"
#include "labeled_btensor.h"

namespace libtensor {

template<typename T>
struct btensor_traits {
	typedef T element_t;
	typedef libvmm::std_allocator<T> allocator_t;
};

template<size_t N, typename T, typename Traits>
class btensor_base : public btensor_i<N, T>, public immutable {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

private:
	block_tensor<N, element_t, allocator_t> m_bt;

public:
	//!	\name Construction and destruction
	//@{
	/**	\brief Constructs a block %tensor using provided information
			about blocks
		\param bi Information about blocks
	 **/
	btensor_base(const bispace<N> &bis) :
		 m_bt(bis.get_bis()) { }

	/**	\brief Constructs a block %tensor using a block %index space
		\param bis Block %index space
	 **/
	btensor_base(const block_index_space<N> &bis) :
		m_bt(bis) { }

	/**	\brief Constructs a block %tensor using information about
			blocks from another block %tensor
		\param bt Another block %tensor
	 **/
	btensor_base(const btensor_i<N, element_t> &bt) :
		m_bt(bt) { }

	/**	\brief Virtual destructor
	 **/
	virtual ~btensor_base() { }
	//@}

	//!	\name Implementation of block_tensor_i<N, T>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N,T>
	//@{
	virtual const symmetry<N, T> &on_req_symmetry() throw(exception);
	virtual void on_req_sym_add_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual void on_req_sym_remove_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual bool on_req_sym_contains_element(
		const symmetry_element_i<N, T> &elem) throw(exception);
	virtual void on_req_sym_clear_elements() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);
	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable();
	//@}
};

/**	\brief User-friendly block %tensor

	\ingroup libtensor_iface
 **/
template<size_t N, typename T = double, typename Traits = btensor_traits<T> >
class btensor : public btensor_base<N, T, Traits> {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

public:
	btensor(const bispace<N> &bi) : btensor_base<N, T, Traits>(bi) { }
	btensor(const block_index_space<N> &bis) :
		btensor_base<N, T, Traits>(bis) { }
	btensor(const btensor_i<N, element_t> &bt) :
		btensor_base<N, T, Traits>(bt) { }
	virtual ~btensor() { }

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<N, T, true> operator()(const letter_expr<N> &expr);

};


template<typename T, typename Traits>
class btensor<1, T, Traits> : public btensor_base<1, T, Traits> {
private:
	typedef typename Traits::element_t element_t;
	typedef typename Traits::allocator_t allocator_t;

public:
	btensor(const bispace<1> &bi) : btensor_base<1, T, Traits>(bi) { }
	btensor(const block_index_space<1> &bis) :
		btensor_base<1, T, Traits>(bis) { }
	btensor(const btensor_i<1, element_t> &bt) :
		btensor_base<1, T, Traits>(bt) { }
	virtual ~btensor() { }

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<1, T, true> operator()(const letter &let);

	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<1, T, true> operator()(const letter_expr<1> &expr);

};


template<size_t N, typename T, typename Traits>
inline const block_index_space<N> &btensor_base<N, T, Traits>::get_bis() const {

	return m_bt.get_bis();
}

template<size_t N, typename T, typename Traits>
const symmetry<N, T> &btensor_base<N, T, Traits>::on_req_symmetry()
	throw(exception) {

	block_tensor_ctrl<N, T> ctrl(m_bt);
	return ctrl.req_symmetry();
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_sym_add_element(elem);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_sym_remove_element(elem);
}

template<size_t N, typename T, typename Traits>
bool btensor_base<N, T, Traits>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	block_tensor_ctrl<N, T> ctrl(m_bt);
	return ctrl.req_sym_contains_element(elem);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_sym_clear_elements() throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_sym_clear_elements();
}

template<size_t N, typename T, typename Traits>
tensor_i<N, T> &btensor_base<N, T, Traits>::on_req_block(const index<N> &idx)
	throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	return ctrl.req_block(idx);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_ret_block(const index<N> &idx)
	throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.ret_block(idx);
}

template<size_t N, typename T, typename Traits>
bool btensor_base<N, T, Traits>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	return ctrl.req_is_zero_block(idx);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_zero_block(const index<N> &idx)
	throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_zero_block(idx);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_zero_all_blocks() throw(exception) {
	block_tensor_ctrl<N, T> ctrl(m_bt);
	ctrl.req_zero_all_blocks();
}

template<size_t N, typename T, typename Traits>
inline void btensor_base<N, T, Traits>::on_set_immutable() {
	m_bt.set_immutable();
}

template<size_t N, typename T, typename Traits>
inline labeled_btensor<N, T, true> btensor<N, T, Traits>::operator()(
	const letter_expr<N> &expr) {

	return labeled_btensor<N, T, true>(*this, expr);
}

template<typename T, typename Traits>
inline labeled_btensor<1, T, true> btensor<1, T, Traits>::operator()(
	const letter &let) {

	return labeled_btensor<1, T, true>(*this, letter_expr<1>(let));
}

template<typename T, typename Traits>
inline labeled_btensor<1, T, true> btensor<1, T, Traits>::operator()(
	const letter_expr<1> &expr) {

	return labeled_btensor<1, T, true>(*this, expr);
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

