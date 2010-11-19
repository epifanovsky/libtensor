#ifndef LIBTENSOR_MAPPED_BTENSOR_H
#define LIBTENSOR_MAPPED_BTENSOR_H

#include <libvmm/vm_allocator.h>
#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/mapped_block_tensor.h"
#include "btensor_i.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"

namespace libtensor {


/**	\brief User-friendly mapped block %tensor

	\ingroup libtensor_iface
 **/
template<size_t N, typename T = double>
class mapped_btensor : public btensor_i<N, T> {
private:
	mapped_block_tensor<N, T> m_bt;
	block_tensor_ctrl<N, T> m_ctrl;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the mapped block %tensor
	 **/
	mapped_btensor(btensor_i<N, T> &bt, const block_index_map_i<N> &bimap,
		const symmetry<N, T> &sym);

	/**	\brief Virtual destructor
	 **/
	virtual ~mapped_btensor() { }

	//@}


	/**	\brief Attaches a label to this %tensor and returns it as a
			labeled %tensor
	 **/
	labeled_btensor<N, T, false> operator()(const letter_expr<N> &expr);


	//!	\name Implementation of block_tensor_i<N, T>
	//@{

	virtual const block_index_space<N> &get_bis() const;

	//@}

protected:
	//!	\name Implementation of block_tensor_i<N, T>
	//@{

	virtual symmetry<N, T> &on_req_symmetry() throw(exception);
	virtual const symmetry<N, T> &on_req_const_symmetry() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_aux_block(const index<N> &idx) throw(exception);
	virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);
	virtual void on_req_sync_on() throw(exception);
	virtual void on_req_sync_off() throw(exception);

	//@}

};


template<size_t N, typename T>
mapped_btensor<N, T>::mapped_btensor(btensor_i<N, T> &bt,
	const block_index_map_i<N> &bimap, const symmetry<N, T> &sym) :

	m_bt(bt, bimap, sym), m_ctrl(m_bt) {

}


template<size_t N, typename T>
labeled_btensor<N, T, false> mapped_btensor<N, T>::operator()(
	const letter_expr<N> &expr) {

	return labeled_btensor<N, T, false>(*this, expr);
}


template<size_t N, typename T>
const block_index_space<N> &mapped_btensor<N, T>::get_bis() const {

	return m_bt.get_bis();
}


template<size_t N, typename T>
symmetry<N, T> &mapped_btensor<N, T>::on_req_symmetry()
	throw(exception) {

	return m_ctrl.req_symmetry();
}


template<size_t N, typename T>
const symmetry<N, T> &mapped_btensor<N, T>::on_req_const_symmetry()
	throw(exception) {

	return m_ctrl.req_const_symmetry();
}


template<size_t N, typename T>
tensor_i<N, T> &mapped_btensor<N, T>::on_req_block(const index<N> &idx)
	throw(exception) {

	return m_ctrl.req_block(idx);
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_ret_block(const index<N> &idx) throw(exception) {

	m_ctrl.ret_block(idx);
}


template<size_t N, typename T>
tensor_i<N, T> &mapped_btensor<N, T>::on_req_aux_block(const index<N> &idx)
	throw(exception) {

	return m_ctrl.req_aux_block(idx);
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	m_ctrl.ret_aux_block(idx);
}


template<size_t N, typename T>
bool mapped_btensor<N, T>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	return m_ctrl.req_is_zero_block(idx);
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	m_ctrl.req_zero_block(idx);
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_req_zero_all_blocks() throw(exception) {

	m_ctrl.req_zero_all_blocks();
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_req_sync_on() throw(exception) {

	m_ctrl.req_sync_on();
}


template<size_t N, typename T>
void mapped_btensor<N, T>::on_req_sync_off() throw(exception) {

	m_ctrl.req_sync_off();
}


} // namespace libtensor

#endif // LIBTENSOR_MAPPED_BTENSOR_H
