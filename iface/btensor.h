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
	virtual size_t on_req_sym_num_orbits() throw(exception);
	virtual orbit<N, T> on_req_sym_orbit(size_t n) throw(exception);
	virtual orbit_iterator<N, T> on_req_orbits() throw(exception);
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

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const bispace_i<N> &bispace)
: m_bt(bispace.get_bis()) {

}

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const block_index_space<N> &bis)
: m_bt(bis) {

}

template<size_t N, typename T, typename Traits>
inline btensor<N, T, Traits>::btensor(const btensor_i<N, element_t> &bt)
: m_bt(bt) {

}

template<size_t N, typename T, typename Traits>
btensor<N, T, Traits>::~btensor() {

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
const symmetry<N, T> &btensor<N, T, Traits>::on_req_symmetry()
	throw(exception) {

	throw_exc("btensor<N, T, Traits>", "on_req_symmetry()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("btensor<N, T, Traits>", "on_req_sym_add_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("btensor<N, T, Traits>", "on_req_sym_remove_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
bool btensor<N, T, Traits>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	throw_exc("btensor<N, T, Traits>", "on_req_sym_contains_element()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_sym_clear_elements() throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_sym_clear_elements()", "NIY");
}

template<size_t N, typename T, typename Traits>
size_t btensor<N, T, Traits>::on_req_sym_num_orbits() throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_sym_num_orbits()", "NIY");
}

template<size_t N, typename T, typename Traits>
orbit<N, T> btensor<N, T, Traits>::on_req_sym_orbit(size_t n) throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_sym_orbit()", "NIY");
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
bool btensor<N, T, Traits>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_is_zero_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_zero_block(const index<N> &idx)
	throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_zero_block()", "NIY");
}

template<size_t N, typename T, typename Traits>
void btensor<N, T, Traits>::on_req_zero_all_blocks() throw(exception) {
	throw_exc("btensor<N, T, Traits>", "on_req_zero_all_blocks()", "NIY");
}

template<size_t N, typename T, typename Traits>
inline void btensor<N, T, Traits>::on_set_immutable() {
	m_bt.set_immutable();
}

} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

