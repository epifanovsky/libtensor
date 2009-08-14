#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <algorithm>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "orbit.h"
#include "symmetry_element_i.h"

namespace libtensor {

/**	\brief Tensor symmetry
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry {
public:
	static const char *k_clazz; //!< Class name

public:
	//!	Symmetry element type
	typedef symmetry_element_i<N, T> symmetry_element_t;

private:
	dimensions<N> m_dims; //!< Block %index %dimensions
	std::vector<symmetry_element_t*> m_elements; //!< Symmetry elements
	mutable std::vector<size_t> m_orbits; //!< Orbits
	mutable bool m_dirty; //!< Indicates that the orbits need to be rebuilt

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


	//!	\name Symmetry elements
	//@{

	/**	\brief Returns the number of %symmetry elements
	 **/
	size_t get_num_elements() const;

	/**	\brief Returns a %symmetry element
		\param n Element number, not to exceed the total number of
			%symmetry elements.
		\throw out_of_bounds If the element number is out of bounds.
	 **/
	const symmetry_element_t &get_element(size_t n) const
		throw(out_of_bounds);

	/**	\brief Adds a %symmetry element to the generating set; does
			nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void add_element(const symmetry_element_t &elem);

	/**	\brief Removes a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void remove_element(const symmetry_element_t &elem);

	/**	\brief Checks whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool contains_element(const symmetry_element_t &elem) const;

	/**	\brief Removes all elements from the generating set
	 **/
	void clear_elements();

	/**	\brief Creates the union of two generating sets
		\param sym Second symmetry.
	 **/
	void element_set_union(const symmetry<N, T> &sym);

	/**	\brief Creates the overlap of two generating sets
		\param sym Second symmetry.
	 **/
	void element_set_overlap(const symmetry<N, T> &sym);

	/**	\brief Adjusts all elements to reflect the %symmetry of a
			permuted %tensor
	 **/
	void permute(const permutation<N> &perm);

	//@}


	//!	\name Orbits
	//@{

	/**	\brief Returns the number of orbits
	 **/
	size_t get_num_orbits() const;

	/**	\brief Return an orbit
		\param n Orbit number.
		\throw out_of_bounds If the orbit number provided is larger
			than the total number of orbits.
	 **/
	orbit<N, T> get_orbit(size_t n) const throw(out_of_bounds);

	/**	\brief Returns whether the element with a given %index is
			canonical
		\throw out_of_bounds If the %index is outside of the symmetry's
			dimensions.
	 **/
	bool is_canonical(const index<N> &idx) const throw(out_of_bounds);

	void get_transf(const index<N> &idx, index<N> &can, transf<N, T> &tr)
		const throw(out_of_bounds);

	//@}

private:
	void remove_all();
	void make_orbits() const;
	bool mark_orbit(const index<N> &idx, std::vector<bool> &lst) const;
	bool find_canonical(const index<N> &idx, std::vector<bool> &lst,
		index<N> &can, transf<N, T> &tr) const;

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

	typename std::vector<symmetry_element_t*>::const_iterator i =
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
size_t symmetry<N, T>::get_num_elements() const {

	return m_elements.size();
}


template<size_t N, typename T>
const symmetry_element_i<N, T> &symmetry<N, T>::get_element(size_t n) const
	throw(out_of_bounds) {

	static const char *method = "get_element(size_t)";

	if(n >= m_elements.size()) {
		throw out_of_bounds("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Element number is out of bounds.");
	}
	return *(m_elements[n]);
}


template<size_t N, typename T>
void symmetry<N, T>::add_element(const symmetry_element_i<N, T> &elem) {

	typename std::vector<symmetry_element_t*>::iterator i =
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
void symmetry<N, T>::remove_element(const symmetry_element_i<N, T> &elem) {

	typename std::vector<symmetry_element_t*>::iterator i =
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
bool symmetry<N, T>::contains_element(const symmetry_element_i<N, T> &elem)
	const {

	typename std::vector<symmetry_element_t*>::const_iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if((*i)->equals(elem)) return true;
		i++;
	}
	return false;
}


template<size_t N, typename T>
void symmetry<N, T>::clear_elements() {

	remove_all();
}


template<size_t N, typename T>
void symmetry<N, T>::element_set_union(const symmetry<N, T> &sym) {

	typename std::vector<symmetry_element_t*>::iterator i =
		sym.m_elements.begin();
	while(i != sym.m_elements.end()) {
		add_element(*i);
		i++;
	}
}


template<size_t N, typename T>
void symmetry<N, T>::element_set_overlap(const symmetry<N, T> &sym) {

	typename std::vector<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if(!sym.contains_element(**i)) {
			symmetry_element_t *ptr = *i;
			*i = NULL;
			delete ptr;
			i = m_elements.erase(i);
			m_dirty = true;
		} else {
			i++;
		}
	}
}


template<size_t N, typename T>
void symmetry<N, T>::permute(const permutation<N> &perm) {

	m_dims.permute(perm);
	typename std::vector<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		(*i)->permute(perm);
		i++;
	}
}


template<size_t N, typename T>
size_t symmetry<N, T>::get_num_orbits() const {

	if(m_dirty) make_orbits();
	return m_orbits.size();
}


template<size_t N, typename T>
orbit<N, T> symmetry<N, T>::get_orbit(size_t n) const throw(out_of_bounds) {

	static const char *method = "get_orbit(size_t)";

	if(m_dirty) make_orbits();
	if(n >= m_orbits.size()) {
		throw out_of_bounds("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Orbit number is out of bounds.");
	}
	return orbit<N, T>(*this, m_orbits[n]);
}


template<size_t N, typename T>
bool symmetry<N, T>::is_canonical(const index<N> &idx) const
	throw(out_of_bounds) {

	if(m_dirty) make_orbits();
	size_t absidx = m_dims.abs_index(idx);
	typename std::vector<size_t>::const_iterator i =
		std::find(m_orbits.begin(), m_orbits.end(), absidx);
	return i != m_orbits.end();
}


template<size_t N, typename T>
void symmetry<N, T>::get_transf(const index<N> &idx, index<N> &can,
	transf<N, T> &tr) const throw(out_of_bounds) {

	tr.reset();
	std::vector<bool> chk(m_dims.get_size(), false);
	find_canonical(idx, chk, can, tr);
	tr.invert();
}


template<size_t N, typename T>
void symmetry<N, T>::remove_all() {

	if(m_elements.empty()) return;
	typename std::vector<symmetry_element_t*>::iterator i =
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
void symmetry<N, T>::make_orbits() const {

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
bool symmetry<N, T>::mark_orbit(
	const index<N> &idx, std::vector<bool> &lst) const {

	size_t absidx = m_dims.abs_index(idx);
	bool allowed = true;
	if(!lst[absidx]) {
		lst[absidx] = true;
		typename std::vector<symmetry_element_t*>::const_iterator
			ielem = m_elements.begin();
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


template<size_t N, typename T>
bool symmetry<N, T>::find_canonical(const index<N> &idx, std::vector<bool> &lst,
	index<N> &can, transf<N, T> &tr) const {

	size_t absidx = m_dims.abs_index(idx);
	if(!lst[absidx]) {
		lst[absidx] = true;
		if(is_canonical(idx)) {
			can = idx;
			return true;
		}
		typename std::vector<symmetry_element_t*>::const_iterator
			ielem = m_elements.begin();
		while(ielem != m_elements.end()) {
			index<N> idx2(idx);
			transf<N, T> tr2;
			(*ielem)->apply(idx2, tr2);
			if(find_canonical(idx2, lst, can, tr2)) {
				tr.transform(tr2);
				return true;
			}
			ielem++;
		}
	}
	return false;
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H
