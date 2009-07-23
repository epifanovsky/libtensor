#ifndef LIBTENSOR_INDEX_H
#define LIBTENSOR_INDEX_H

#include <iostream>
#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

template<size_t N> class index;
template<size_t N> std::ostream &operator<<(std::ostream &os,
	const index<N> &i);

/**	\brief Index of a single %tensor element
	\tparam N Index order.

	An %index is a sequence of integers that identifies a single %tensor
	element. The number of integers in the sequence (the order of the
	%index) agrees with the number of %tensor %dimensions. Each integer
	of the sequence gives the position of the element along each dimension.


	<b>Creation of indexes</b>

	A newly created %index object points at the first %tensor element, i.e.
	has zeros along each dimension. To modify the %index, at() or
	operator[] can be used:
	\code
	index<2> i, j;
	i[0] = 2; i[1] = 3;
	j.at(0) = 2; j.at(1) = 3; // operator[] and at() are identical
	\endcode


	<b>Comparison methods</b>

	Two %index objects can be compared using equals(), which returns true
	two indexes identify the same element in a %tensor, and false otherwise:
	\code
	index<2> i, j;
	bool b;
	i[0] = 2; i[1] = 3;
	j[0] = 2; j[1] = 3;
	b = i.equals(j); // b == true
	j[0] = 3; j[1] = 2;
	b = i.equals(j); // b == false
	\endcode

	For convenience, operator== and operator!= are overloaded for indexes.
	Continuing the above code example,
	\code
	j[0] = 2; j[1] = 3;
	if(i == j) {
		// code here will be executed
	}
	if(i != j) {
		// code here will not be executed
	}
	\endcode

	Two non-equal indexes can be put in an ordered sequence using the
	defined comparison operation: each %index element is compared according
	to its seniority, the first element being the most senior, and the
	last element being junior. The comparison is performed with the less()
	method or overloaded operator< and operator>.
	\code
	i[0] = 2; i[1] = 3;
	j[0] = 2; j[1] = 4;
	if(i.less(j)) {
		// code here will be executed
	}
	j[0] = 1; j[1] = 4;
	if(i.less(j)) {
		// code here will not be executed
	}
	\endcode


	<b>Output to a stream</b>

	To print the current value of the %index to an output stream, the
	overloaded operator<< can be used.


	<b>Exceptions</b>

	Methods and operators that require an input position or %index may
	throw an out_of_bounds %exception if supplied input falls out of
	allowable range.


	\ingroup libtensor_core
**/
template<size_t N>
class index {
	friend std::ostream &operator<< <N>(std::ostream &os, const index<N> &i);

private:
	size_t m_idx[N]; //!< Index elements

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor creates an %index with all zeros
	 **/
	index();

	/**	\brief Copies the %index from another instance
		\param idx Another %index.
	 **/
	index(const index<N> &idx);

	//@}


	//!	\name Access to %index elements, manipulations
	//@{

	/**	\brief Returns the reference to an element at a given position
			(r-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	size_t &at(size_t pos) throw(out_of_bounds);

	/**	\brief Returns an element at a given position (l-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	size_t at(size_t pos) const throw(out_of_bounds);

	/**	\brief Permutes the elements of the %index
		\param p Permutation.
		\return Reference to this %index.
	 **/
	index<N> &permute(const permutation<N> &perm);

	//@}


	//!	\name Comparison
	//@{

	/**	\brief Checks if two indices are equal

		Returns true if the indices are equal, false otherwise.
	 **/
	bool equals(const index<N> &idx) const;

	/**	\brief Compares two indexes
		\return true if this %index is smaller than the other one,
			false otherwise.
	 **/
	bool less(const index<N> &idx) const;

	//@}


	//!	\name Overloaded operators
	//@{

	/**	\brief Returns the reference to an element at a given position
			(r-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	size_t &operator[](size_t pos) throw(out_of_bounds);

	/**	\brief Returns an element at a given position (l-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	size_t operator[](size_t pos) const throw(out_of_bounds);

	//@}

};


template<size_t N>
inline index<N>::index() {

	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++) m_idx[i] = 0;
}

template<size_t N>
inline index<N>::index(const index<N> &idx) {

	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++) m_idx[i] = idx.m_idx[i];
}

template<size_t N>
inline size_t &index<N>::at(size_t pos) throw(out_of_bounds) {

#ifdef LIBTENSOR_DEBUG
	if(pos >= N) {
		throw out_of_bounds("libtensor", "index<N>", "at(size_t)",
			__FILE__, __LINE__, "pos");
	}
#endif // LIBTENSOR_DEBUG
	return m_idx[pos];
}

template<size_t N>
inline size_t index<N>::at(size_t pos) const throw(out_of_bounds) {

#ifdef LIBTENSOR_DEBUG
	if(pos >= N) {
		throw out_of_bounds("libtensor", "index<N>", "at(size_t) const",
			__FILE__, __LINE__, "pos");
	}
#endif // LIBTENSOR_DEBUG
	return m_idx[pos];
}

template<size_t N>
inline index<N> &index<N>::permute(const permutation<N> &perm) {

	perm.apply(m_idx);
	return *this;
}


template<size_t N>
inline bool index<N>::equals(const index<N> &idx) const {

	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++)
		if(m_idx[i] != idx.m_idx[i]) return false;
	return true;
}

template<size_t N>
inline bool index<N>::less(const index<N> &idx) const {

	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++) {
		if(m_idx[i] < idx.m_idx[i]) return true;
		if(m_idx[i] > idx.m_idx[i]) return false;
	}
	return false;
}

template<size_t N>
inline size_t &index<N>::operator[](size_t pos) throw(out_of_bounds) {

	return at(pos);
}

template<size_t N>
inline size_t index<N>::operator[](size_t pos) const
	throw(out_of_bounds) {

	return at(pos);
}

/**	\brief Prints out the index to an output stream

	\ingroup libtensor
**/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const index<N> &i) {
	os << "[";
	for(size_t j=0; j<N-1; j++) os << i.m_idx[j] << ",";
	os << i.m_idx[N-1];
	os << "]";
	return os;
}

} // namespace libtensor

#endif // LIBTENSOR_INDEX_H

