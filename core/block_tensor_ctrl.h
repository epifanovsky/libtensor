#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "core/index.h"

namespace libtensor {

/**	\brief Block %tensor control
	\tparam N Block %tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
**/
template<size_t N, typename T>
class block_tensor_ctrl {
private:
	block_tensor_i<N, T> &m_bt; //!< Controlled block %tensor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the control object
	 **/
	block_tensor_ctrl(block_tensor_i<N, T> &bt);

	/**	\brief Destroys the control object
	 **/
	~block_tensor_ctrl();

	//@}


	//!	\name Symmetry events
	//@{

	/**	\brief Request to obtain the constant reference to the %tensor's
			%symmetry
	 **/
	const symmetry<N, T> &req_symmetry() throw(exception);

	/**	\brief Request to return the number of %symmetry elements
	 **/
	//size_t req_sym_num_elem() throw(exception);

	/**	\brief Request to return a %symmetry element
		\param n Element number, not to exceed the total number of
			%symmetry elements.
		\throw out_of_bounds If the element number is out of bounds.
	 **/
	//const symmetry_element_i<N, T> &req_sym_elem(size_t n) throw(exception);

	/**	\brief Request to add a %symmetry element to the generating set;
			does nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void req_sym_add_element(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request to remove a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void req_sym_remove_element(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool req_sym_contains_element(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request to clear the generating set
	 **/
	void req_sym_clear_elements() throw(exception);

	//@}

	//!	\name Block events
	//@{
	tensor_i<N, T> &req_block(const index<N> &idx) throw(exception);
	void ret_block(const index<N> &idx) throw(exception);
	bool req_is_zero_block(const index<N> &idx) throw(exception);
	void req_zero_block(const index<N> &idx) throw(exception);
	void req_zero_all_blocks() throw(exception);
	//@}
};

template<size_t N, typename T>
inline block_tensor_ctrl<N, T>::block_tensor_ctrl(block_tensor_i<N, T> &bt) :
	m_bt(bt) {
}

template<size_t N, typename T>
block_tensor_ctrl<N, T>::~block_tensor_ctrl() {
}


template<size_t N, typename T>
inline const symmetry<N, T> &block_tensor_ctrl<N, T>::req_symmetry()
	throw(exception) {

	return m_bt.on_req_symmetry();
}


template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_sym_add_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	m_bt.on_req_sym_add_element(elem);
}


template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_sym_remove_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	m_bt.on_req_sym_remove_element(elem);
}


template<size_t N, typename T>
inline bool block_tensor_ctrl<N, T>::req_sym_contains_element(
	const symmetry_element_i<N, T> &elem) throw(exception) {

	return m_bt.on_req_sym_contains_element(elem);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_sym_clear_elements() throw(exception) {

	m_bt.on_req_sym_clear_elements();
}

template<size_t N, typename T>
inline tensor_i<N, T> &block_tensor_ctrl<N, T>::req_block(const index<N> &idx)
	throw(exception) {

	return m_bt.on_req_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::ret_block(const index<N> &idx)
	throw(exception) {

	return m_bt.on_ret_block(idx);
}

template<size_t N, typename T>
inline bool block_tensor_ctrl<N, T>::req_is_zero_block(const index<N> &idx)
	throw(exception) {

	return m_bt.on_req_is_zero_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_zero_block(const index<N> &idx)
	throw(exception) {

	m_bt.on_req_zero_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_zero_all_blocks() throw(exception) {

	m_bt.on_req_zero_all_blocks();
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

