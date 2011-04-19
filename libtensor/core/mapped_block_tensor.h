#ifndef LIBTENSOR_MAPPED_BLOCK_TENSOR_H
#define LIBTENSOR_MAPPED_BLOCK_TENSOR_H

#include "block_index_map.h"
#include "block_tensor_ctrl.h"
#include "block_tensor_i.h"
#include "symmetry.h"
#include "../symmetry/so_copy.h"

namespace libtensor {


template<size_t N, typename T>
class mapped_block_tensor : public block_tensor_i<N, T> {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, T> &m_bt; //!< Underlying block %tensor
	block_tensor_ctrl<N, T> m_ctrl; //!< Control to the underlying
	block_index_map<N> m_bimap; //!< Block %index map
	dimensions<N> m_bidims_from; //!< Source block %index dims
	dimensions<N> m_bidims_to; //!< Target block %index dims
	symmetry<N, T> m_sym; //!< Symmetry

public:
	//!	\name Construction and destruction
	//@{

	mapped_block_tensor(block_tensor_i<N, T> &bt,
		const block_index_map_i<N> &bimap,
		const symmetry<N, T> &sym);

	/**	\brief Virtual destructor
	 **/
	~mapped_block_tensor() { }

	//@}


	//!	\name Implementation of block_tensor_i<N, T>
	//@{

	virtual const block_index_space<N> &get_bis() const;

	//@}

protected:
	//!	\name Implementation of block_tensor_i<N, T>
	//@{

	virtual const symmetry<N, T> &on_req_const_symmetry() throw(exception);
	virtual symmetry<N, T> &on_req_symmetry() throw(exception);
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
const char *mapped_block_tensor<N, T>::k_clazz = "mapped_block_tensor<N, T>";


template<size_t N, typename T>
mapped_block_tensor<N, T>::mapped_block_tensor(block_tensor_i<N, T> &bt,
	const block_index_map_i<N> &bimap, const symmetry<N, T> &sym) :

	m_bt(bt), m_ctrl(m_bt), m_bimap(bimap),
	m_bidims_from(m_bimap.get_bis_from().get_block_index_dims()),
	m_bidims_to(m_bimap.get_bis_to().get_block_index_dims()),
	m_sym(bimap.get_bis_from()) {

	static const char *method =
		"mapped_block_tensor(block_tensor_i<N, T>&, "
		"const block_index_map_i<N>&, const symmetry<N, T>&)";

#ifdef LIBTENSOR_DEBUG
	if(!m_bimap.get_bis_to().equals(m_bt.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"bimap[to]");
	}
	if(!m_bimap.get_bis_from().equals(sym.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"bimap[from]");
	}
#endif // LIBTENSOR_DEBUG

	so_copy<N, T>(sym).perform(m_sym);
}


template<size_t N, typename T>
const block_index_space<N> &mapped_block_tensor<N, T>::get_bis() const {

	return m_bimap.get_bis_from();
}


template<size_t N, typename T>
const symmetry<N, T> &mapped_block_tensor<N, T>::on_req_const_symmetry()
	throw(exception) {

	return m_sym;
}


template<size_t N, typename T>
symmetry<N, T> &mapped_block_tensor<N, T>::on_req_symmetry() throw(exception) {

	static const char *method = "on_req_symmetry()";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T>
tensor_i<N, T> &mapped_block_tensor<N, T>::on_req_block(const index<N> &idx)
	throw(exception) {

	index<N> idx2;
	m_bimap.get_map(idx, idx2);
	return m_ctrl.req_block(idx2);
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_ret_block(const index<N> &idx)
	throw(exception) {

	index<N> idx2;
	m_bimap.get_map(idx, idx2);
	m_ctrl.ret_block(idx2);
}


template<size_t N, typename T>
tensor_i<N, T> &mapped_block_tensor<N, T>::on_req_aux_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T>
bool mapped_block_tensor<N, T>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	if(!m_bimap.map_exists(idx)) return true;
	index<N> idx2;
	m_bimap.get_map(idx, idx2);
	return m_ctrl.req_is_zero_block(idx2);
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_zero_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_req_zero_all_blocks() throw(exception) {

	static const char *method = "on_req_zero_all_blocks()";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_req_sync_on() throw(exception) {

	m_ctrl.req_sync_on();
}


template<size_t N, typename T>
void mapped_block_tensor<N, T>::on_req_sync_off() throw(exception) {

	m_ctrl.req_sync_off();
}


} // namespace libtensor

#endif // LIBTENSOR_MAPPED_BLOCK_TENSOR_H
