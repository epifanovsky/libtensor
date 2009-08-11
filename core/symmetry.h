#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <list>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "orbit.h"
#include "symmetry_element_i.h"

namespace libtensor {

template<size_t N, typename T> class symmetry_ctrl;

/**	\brief Tensor symmetry
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry {
	friend class symmetry_ctrl<N, T>;

public:
	//!	Symmetry element type
	typedef symmetry_element_i<N, T> symmetry_element_t;

public:
	static const char *k_clazz; //!< Class name

private:
	dimensions<N> m_dims; //!< Block %index %dimensions
	std::list<symmetry_element_t*> m_elements; //!< Symmetry elements
	std::vector<size_t> m_orbits; //!< Orbits
	bool m_dirty; //!< Indicates that the orbits need to be rebuilt

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates %symmetry using given %dimensions
		\param dims Block %index %dimensions.
	 **/
	symmetry(const dimensions<N> &dims);

	/**	\brief Copy constructor
		\param sym Another %symmetry object.
	 **/
	symmetry(const symmetry<N, T> &sym);

	/**	\brief Destructor
	 **/
	~symmetry();

	//@}

protected:
	//!	\name Event handlers (see symmetry_ctrl<N, T>)
	//@{

	/**	\brief Request to return the number of %symmetry elements
	 **/
	size_t on_req_num_elem() throw(exception);

	/**	\brief Request to return a %symmetry element
		\param n Element number, not to exceed the total number of
			%symmetry elements.
		\throw out_of_bounds If the element number is out of bounds.
	 **/
	const symmetry_element_t &on_req_elem(size_t n) throw(exception);

	/**	\brief Request to add a %symmetry element to the generating set;
			does nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void on_req_insert_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request to remove a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void on_req_remove_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool on_req_contains_elem(const symmetry_element_t &elem)
		throw(exception);

	/**	\brief Request to clear the generating set
	 **/
	void on_req_clear_elem() throw(exception);

	/**	\brief Request to return the number of orbits
	 **/
	size_t on_req_num_orbits() throw(exception);

	/**	\brief Request to return an orbit
		\param n Orbit number.
		\throw out_of_bounds If the orbit number provided is larger
			than the total number of orbits.
	 **/
	orbit<N, T> on_req_orbit(size_t n) throw(exception);

	//@}

private:
	void remove_all();
	void make_orbits();
	bool mark_orbit(const index<N> &idx, std::vector<bool> &lst);

};


template<size_t N, typename T>
const char *symmetry<N, T>::k_clazz = "symmetry<N, T>";


template<size_t N, typename T>
inline symmetry<N, T>::symmetry(const dimensions<N> &dims)
: m_dims(dims), m_dirty(true) {

}


template<size_t N, typename T>
symmetry<N, T>::symmetry(const symmetry<N, T> &sym)
: m_dims(sym.m_dims), m_dirty(true) {

	typename std::list<symmetry_element_t*>::iterator i =
		sym.m_elements.begin();
	while(i != sym.m_elements.end()) {
		m_elements.push_back((*i)->clone());
		i++;
	}
}


template<size_t N, typename T>
symmetry<N, T>::~symmetry() {

	remove_all();
}


template<size_t N, typename T>
size_t symmetry<N, T>::on_req_num_elem() throw(exception) {

	return m_elements.size();
}


template<size_t N, typename T>
const symmetry_element_i<N, T> &symmetry<N, T>::on_req_elem(size_t n)
	throw(exception) {

	static const char *method = "on_req_elem(size_t)";

	if(n >= m_elements.size()) {
		throw out_of_bounds("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Element number is out of bounds.");
	}
	return *(m_elements[n]);
}


template<size_t N, typename T>
void symmetry<N, T>::on_req_insert_elem(const symmetry_element_i<N, T> &elem)
	throw(exception) {

	typename std::list<symmetry_element_t*>::iterator i =
		m_elements.begin();
	bool found = false;
	while(i != m_elements.end()) {
		if((*i)->equals(elem)) {
			found = true;
			break;
		}
		i++;
	}
	if(!found) {
		m_elements.push_back(elem.clone());
		m_dirty = true;
	}
}


template<size_t N, typename T>
void symmetry<N, T>::on_req_remove_elem(const symmetry_element_i<N, T> &elem)
	throw(exception) {

	typename std::list<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if((*i)->equals(elem)) {
			symmetry_element_t *ptr = *i;
			*i = NULL;
			delete ptr;
			m_dirty = true;
			break;
		}
		i++;
	}
}


template<size_t N, typename T>
bool symmetry<N, T>::on_req_contains_elem(const symmetry_element_i<N, T> &elem)
	throw(exception) {

	typename std::list<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if((*i)->equals(elem)) return true;
		i++;
	}
	return false;
}


template<size_t N, typename T>
void symmetry<N, T>::on_req_clear_elem() throw(exception) {

	remove_all();
}


template<size_t N, typename T>
size_t symmetry<N, T>::on_req_num_orbits() throw(exception) {

	if(m_dirty) make_orbits();
	return m_orbits.size();
}


template<size_t N, typename T>
orbit<N, T> symmetry<N, T>::on_req_orbit(size_t n) throw(exception) {

	static const char *method = "on_req_orbit(size_t)";

	if(m_dirty) make_orbits();
	if(n >= m_orbits.size()) {
		throw out_of_bounds("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Orbit number is out of bounds.");
	}
	return orbit<N, T>(*this, m_orbits[n]);
}


template<size_t N, typename T>
void symmetry<N, T>::remove_all() {

	if(m_elements.empty()) return;
	typename std::list<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		symmetry_element_t *ptr = *i;
		*i = NULL;
		delete ptr;
		i++;
	}
	m_elements.clear();
	m_dirty = true;
}


template<size_t N, typename T>
void symmetry<N, T>::make_orbits() {

	m_orbits.clear();

	std::vector<bool> chk(m_dims.get_size(), false);
	index<N> idx;
	do {
		size_t absidx = m_dims.abs_index(idx);
		if(!chk[absidx]) {
			if(mark_orbit(idx, chk))
				m_orbits.push_back(absidx);
		}
	} while(m_dims.inc_index(idx));

	m_dirty = false;
}


template<size_t N, typename T>
bool symmetry<N, T>::mark_orbit(const index<N> &idx, std::vector<bool> &lst) {

	size_t absidx = m_dims.abs_index(idx);
	bool allowed = true;
	if(!lst[absidx]) {
		lst[absidx] = true;
		typename std::list<symmetry_element_t*>::iterator ielem =
			m_elements.begin();
		while(ielem != m_elements.end()) {
			allowed = allowed && (*ielem)->is_allowed(idx);
			index<N> idx2(idx);
			(*ielem)->apply(idx2);
			allowed = allowed && mark_orbit(idx2, lst);
			ielem++;
		}
	}
	return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H
