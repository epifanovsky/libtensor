#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "block_map.h"
#include "block_tensor_i.h"
#include "immutable.h"
#include "orbit_list.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator.

	\ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc>
class block_tensor : public block_tensor_i<N, T>, public immutable {
public:
	static const char *k_clazz; //!< Class name

public:
	block_index_space<N> m_bis; //!< Block %index space
	dimensions<N> m_bidims; //!< Block %index %dimensions
	symmetry<N, T> m_symmetry; //!< Block %tensor symmetry
	block_map<N, T, Alloc> m_map; //!< Block map

public:
	//!	\name Construction and destruction
	//@{
	block_tensor(const block_index_space<N> &bis);
	block_tensor(const block_tensor<N, T, Alloc> &bt);
	virtual ~block_tensor();
	//@}

	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
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


template<size_t N, typename T, typename Alloc>
const char *block_tensor<N, T, Alloc>::k_clazz = "block_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_index_space<N> &bis)
: m_bis(bis), m_bidims(bis.get_block_index_dims()), m_symmetry(m_bidims) {

}


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_tensor<N, T, Alloc> &bt)
: m_bis(bt.get_bis()), m_bidims(bt.m_bidims), m_symmetry(bt.m_symmetry) {

}


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::~block_tensor() {

}


template<size_t N, typename T, typename Alloc>
const block_index_space<N> &block_tensor<N, T, Alloc>::get_bis()
	const {

	return m_bis;
}


template<size_t N, typename T, typename Alloc>
const symmetry<N, T> &block_tensor<N, T, Alloc>::on_req_symmetry()
	throw(exception) {

	return m_symmetry;
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	static const char *method =
		"on_req_sym_add_element(const symmetry_element_i<N, T>&)";

	if(is_immutable()) {
		throw immut_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}

	m_symmetry.add_element(elem);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	static const char *method =
		"on_req_sym_remove_element(const symmetry_element_i<N, T>&)";

	if(is_immutable()) {
		throw immut_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}

	m_symmetry.remove_element(elem);
}


template<size_t N, typename T, typename Alloc>
bool block_tensor<N, T, Alloc>::on_req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	return m_symmetry.contains_element(elem);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_sym_clear_elements() throw(exception) {

	static const char *method = "on_req_sym_clear_elements()";

	if(is_immutable()) {
		throw immut_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}

	m_symmetry.clear_elements();
}

template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &block_tensor<N, T, Alloc>::on_req_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_block(const index<N>&)";

	bool canonical = false;
	orbit_list<N, T> orblst(m_symmetry);
	typename orbit_list<N, T>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		if(idx.equals(*iorbit)) {
			canonical = true;
			break;
		}
	}
	if(!canonical) {
		throw symmetry_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	size_t absidx = m_bidims.abs_index(idx);
	if(!m_map.contains(absidx)) {
		dimensions<N> blkdims = m_bis.get_block_dims(idx);
		m_map.create(absidx, blkdims);
	}
	return m_map.get(absidx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx)
	throw(exception) {

}


template<size_t N, typename T, typename Alloc>
bool block_tensor<N, T, Alloc>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_is_zero_block(const index<N>&)";

	bool canonical = false;
	orbit_list<N, T> orblst(m_symmetry);
	typename orbit_list<N, T>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		if(idx.equals(*iorbit)) {
			canonical = true;
			break;
		}
	}
	if(!canonical) {
		throw symmetry_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	size_t absidx = m_bidims.abs_index(idx);
	return !m_map.contains(absidx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_zero_block(const index<N>&)";

	if(is_immutable()) {
		throw immut_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}
	bool canonical = false;
	orbit_list<N, T> orblst(m_symmetry);
	typename orbit_list<N, T>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {
		if(idx.equals(*iorbit)) {
			canonical = true;
			break;
		}
	}
	if(!canonical) {
		throw symmetry_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	size_t absidx = m_bidims.abs_index(idx);
	m_map.remove(absidx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_zero_all_blocks() throw(exception) {

	static const char *method = "on_req_zero_all_blocks()";

	if(is_immutable()) {
		throw immut_violation("libtensor", k_clazz, method,
			__FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}
	m_map.clear();
}


template<size_t N, typename T, typename Alloc>
inline void block_tensor<N, T, Alloc>::on_set_immutable() {

	m_map.set_immutable();
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H
