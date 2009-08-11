#ifndef LIBTENSOR_SYMMETRY_CTRL_H
#define LIBTENSOR_SYMMETRY_CTRL_H

#include "defs.h"
#include "exception.h"
#include "symmetry.h"

namespace libtensor {

template<size_t N, typename T>
class symmetry_ctrl {
public:
	typedef typename symmetry<N, T>::symmetry_element_t
		symmetry_element_t; //!< Symmetry element type

private:
	symmetry<N, T> &m_sym;

public:
	//!	\name Construction and destruction
	//@{

	symmetry_ctrl(symmetry<N, T> &sym);

	//@}

	//!	\name Symmetry events
	//@{

	/**	\brief Request to return the number of %symmetry elements
	 **/
	size_t req_num_elem() throw(exception);

	/**	\brief Request to return a %symmetry element
		\param n Element number, not to exceed the total number of
			%symmetry elements.
		\throw out_of_bounds If the element number is out of bounds.
	 **/
	const symmetry_element_i<N, T> &req_elem(size_t n) throw(exception);

	/**	\brief Request to add a %symmetry element to the generating set;
			does nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void req_insert_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request to remove a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void req_remove_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool req_contains_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request to clear the generating set
	 **/
	void req_clear_elem() throw(exception);

	/**	\brief Request to return the number of orbits
	 **/
	size_t req_num_orbits() throw(exception);

	/**	\brief Request to return an orbit
		\param n Orbit number.
		\throw out_of_bounds If the orbit number provided is larger
			than the total number of orbits.
	 **/
	orbit<N, T> req_orbit(size_t n) throw(exception);

	//@}

};


template<size_t N, typename T>
inline symmetry_ctrl<N, T>::symmetry_ctrl(symmetry<N, T> &sym) : m_sym(sym) {

}


template<size_t N, typename T>
inline size_t symmetry_ctrl<N, T>::req_num_elem() throw(exception) {

	return m_sym.on_req_num_elem();
}


template<size_t N, typename T>
inline const symmetry_element_i<N, T> &symmetry_ctrl<N, T>::req_elem(size_t n)
	throw(exception) {

	return m_sym.on_req_elem(n);
}


template<size_t N, typename T>
inline void symmetry_ctrl<N, T>::req_insert_elem(const symmetry_element_t &elem)
	throw(exception) {

	m_sym.on_req_insert_elem(elem);
}


template<size_t N, typename T>
inline void symmetry_ctrl<N, T>::req_remove_elem(const symmetry_element_t &elem)
	throw(exception) {

	m_sym.on_req_remove_elem(elem);
}


template<size_t N, typename T>
inline bool symmetry_ctrl<N, T>::req_contains_elem(
	const symmetry_element_t &elem) throw(exception) {

	return m_sym.on_req_contains_elem(elem);
}


template<size_t N, typename T>
inline void symmetry_ctrl<N, T>::req_clear_elem() throw(exception) {

	m_sym.on_req_clear_elem();
}


template<size_t N, typename T>
inline size_t symmetry_ctrl<N, T>::req_num_orbits() throw(exception) {

	return m_sym.on_req_num_orbits();
}


template<size_t N, typename T>
inline orbit<N, T> symmetry_ctrl<N, T>::req_orbit(size_t n) throw(exception) {

	return m_sym.on_req_orbit(n);
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_CTRL_H
