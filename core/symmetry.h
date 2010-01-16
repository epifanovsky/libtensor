#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <algorithm>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "dimensions.h"
#include "symmetry_element_iex.h"

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
	typedef symmetry_element_iex<N, T> symmetry_element_t;

	//!	Iterator
	typedef typename std::vector<symmetry_element_t*>::const_iterator
		iterator;

private:
	block_index_space<N> m_bis; //!< Block %index space
	std::vector<symmetry_element_t*> m_elements; //!< Symmetry elements

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates %symmetry using given %dimensions
		\param dims Block %index %dimensions.
	 **/
	explicit symmetry(const block_index_space<N> &bis);

	/**	\brief Copy constructor
		\param sym Another %symmetry object.
	 **/
	symmetry(const symmetry<N, T> &sym);

	/**	\brief Destructor
	 **/
	~symmetry();

	//@}

	const block_index_space<N> &get_bis() const;


	//!	\name Symmetry elements
	//@{

	/**	\brief Returns the number of %symmetry elements
	 **/
	size_t get_num_elements() const;

	/**	\brief Adds a %symmetry element to the generating set; does
			nothing if the element is already in the set
	`	\param elem Symmetry element.
	 **/
	void add_element(const symmetry_element_i<N, T> &elem);

	/**	\brief Removes a %symmetry element from the generating
			set; does nothing if the element is not in the set
		\param elem Symmetry element.
	 **/
	void remove_element(const symmetry_element_i<N, T> &elem);

	/**	\brief Checks whether the generating set of the %symmetry
			contains a given element
		\param elem Symmetry element.
	 **/
	bool contains_element(const symmetry_element_i<N, T> &elem) const;

	/**	\brief Removes all elements from the generating set
	 **/
	void clear_elements();

	/**	\brief Creates the union of two generating sets
		\param sym Second symmetry.
	 **/
	void set_union(const symmetry<N, T> &sym);

	/**	\brief Creates the intersection of two generating sets
		\param sym Second symmetry.
	 **/
	void set_intersection(const symmetry<N, T> &sym);

	/**	\brief Returns true if two symmetries consist of identical
			generating set elements
		\param sym Another symmetry.
	 **/
	bool equals(const symmetry<N, T> &sym) const;

	/**	\brief Adjusts all elements to reflect the %symmetry of a
			permuted %tensor
	 **/
	void permute(const permutation<N> &perm);

	//@}


	//!	\name Iterators
	//@{

	iterator begin() const;
	iterator end() const;
//	iterator begin(const char *);
//	iterator end(const char *);
//	const char *get_type(iterator &i);

	/**	\brief Returns a %symmetry element
		\param iter Iterator.
		\throw out_of_bounds If the iterator is out of bounds.
	 **/
	const symmetry_element_i<N, T> &get_element(iterator &i) const
		throw(out_of_bounds);

	//@}

private:
	void remove_all();
	symmetry(const dimensions<N>&);

};


template<size_t N, typename T>
const char *symmetry<N, T>::k_clazz = "symmetry<N, T>";


template<size_t N, typename T>
inline symmetry<N, T>::symmetry(const block_index_space<N> &bis) : m_bis(bis) {

}


template<size_t N, typename T>
symmetry<N, T>::symmetry(const symmetry<N, T> &sym) : m_bis(sym.m_bis) {

	typename std::vector<symmetry_element_t*>::const_iterator i =
		sym.m_elements.begin();
	while(i != sym.m_elements.end()) {
		m_elements.push_back(dynamic_cast<symmetry_element_t*>(
			(*i)->clone()));
		i++;
	}
}


template<size_t N, typename T>
symmetry<N, T>::~symmetry() {

	remove_all();
}


template<size_t N, typename T>
const block_index_space<N> &symmetry<N, T>::get_bis() const {

	return m_bis;
}


template<size_t N, typename T>
size_t symmetry<N, T>::get_num_elements() const {

	return m_elements.size();
}


//template<size_t N, typename T>
//const symmetry_element_i<N, T> &symmetry<N, T>::get_element(size_t n) const
//	throw(out_of_bounds) {
//
//	static const char *method = "get_element(size_t)";
//
//	if(n >= m_elements.size()) {
//		throw out_of_bounds("libtensor", k_clazz, method, __FILE__,
//			__LINE__, "Element number is out of bounds.");
//	}
//	return *(m_elements[n]);
//}


template<size_t N, typename T>
void symmetry<N, T>::add_element(const symmetry_element_i<N, T> &elem0) {

	static const char *method =
		"add_element(const symmetry_element_i<N, T>&)";

	const symmetry_element_iex<N, T> &elem =
		dynamic_cast< const symmetry_element_iex<N, T>& >(elem0);
	if(!elem.is_valid_bis(m_bis)) {
		throw symmetry_violation(g_ns, k_clazz, method, __FILE__,
			__LINE__, "Symmetry element is not applicable to "
			"the block index space.");
	}
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
	if(!found) m_elements.push_back(dynamic_cast<symmetry_element_t*>(
		elem.clone()));
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
void symmetry<N, T>::set_union(const symmetry<N, T> &sym) {

	typename std::vector<symmetry_element_t*>::const_iterator i =
		sym.m_elements.begin();
	while(i != sym.m_elements.end()) {
		add_element(**i);
		i++;
	}
}


template<size_t N, typename T>
void symmetry<N, T>::set_intersection(const symmetry<N, T> &sym) {

	typename std::vector<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if(!sym.contains_element(**i)) {
			symmetry_element_t *ptr = *i;
			*i = NULL;
			delete ptr;
			i = m_elements.erase(i);
		} else {
			i++;
		}
	}
}


template<size_t N, typename T>
bool symmetry<N, T>::equals(const symmetry<N, T> &sym) const {

	if(get_num_elements() != sym.get_num_elements()) return false;
	typename std::vector<symmetry_element_t*>::const_iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		if(!sym.contains_element(**i)) return false;
		i++;
	}
	return true;
}


template<size_t N, typename T>
void symmetry<N, T>::permute(const permutation<N> &perm) {

	m_bis.permute(perm);
	typename std::vector<symmetry_element_t*>::iterator i =
		m_elements.begin();
	while(i != m_elements.end()) {
		(*i)->permute(perm);
		i++;
	}
}


template<size_t N, typename T>
inline typename symmetry<N, T>::iterator symmetry<N, T>::begin() const {

	return m_elements.begin();
}


template<size_t N, typename T>
inline typename symmetry<N, T>::iterator symmetry<N, T>::end() const {

	return m_elements.end();
}


template<size_t N, typename T>
inline const symmetry_element_i<N, T> &symmetry<N, T>::get_element(iterator &i)
	const throw(out_of_bounds) {

	if(i == m_elements.end()) {
		throw out_of_bounds(g_ns, k_clazz, "get_element(iterator&)",
			__FILE__, __LINE__, "Iterator is out of bounds.");
	}
	//return *(i->second);
	return **i;
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
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H
