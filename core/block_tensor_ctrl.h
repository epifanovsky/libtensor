#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "index.h"

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

	/**	\brief Request to return the number of %symmetry elements
	 **/
	size_t req_sym_num_elem() throw(exception);

	/**	\brief Request to return a %symmetry element
		\param n Element number, not to exceed the total number of
			%symmetry elements.
		\throw out_of_bounds If the element number is out of bounds.
	 **/
	const symmetry_element_i<N, T> &req_sym_elem(size_t n) throw(exception);

	/**	\brief Request to add a %symmetry element to the generating set;
			does nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void req_sym_insert_elem(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request to remove a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void req_sym_remove_elem(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool req_sym_contains_elem(const symmetry_element_i<N, T> &elem)
		throw(exception);

	/**	\brief Request to clear the generating set
	 **/
	void req_sym_clear_elem() throw(exception);

	/**	\brief Request to return the number of orbits
	 **/
	size_t req_sym_num_orbits() throw(exception);

	/**	\brief Request to return an orbit
		\param n Orbit number.
		\throw out_of_bounds If the orbit number provided is larger
			than the total number of orbits.
	 **/
	orbit<N, T> req_sym_orbit(size_t n) throw(exception);

	//@}

	//!	\name Block events
	//@{
	//symmetry<N, T> &req_symmetry() throw(exception);
	tensor_i<N, T> &req_block(const index<N> &idx) throw(exception);
	void ret_block(const index<N> &idx) throw(exception);
	void req_zero_block(const index<N> &idx) throw(exception);
	//@}
};

template<size_t N, typename T>
inline block_tensor_ctrl<N, T>::block_tensor_ctrl(block_tensor_i<N, T> &bt) :
	m_bt(bt) {
}

template<size_t N, typename T>
block_tensor_ctrl<N, T>::~block_tensor_ctrl() {
}
/*
template<size_t N, typename T>
inline symmetry<N, T> &block_tensor_ctrl<N, T>::req_symmetry()
	throw(exception) {

	return m_bt.on_req_symmetry();
}
*/
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
inline void block_tensor_ctrl<N, T>::req_zero_block(const index<N> &idx)
	throw(exception) {

	return m_bt.on_req_zero_block(idx);
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

