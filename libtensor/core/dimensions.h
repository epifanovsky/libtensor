#ifndef LIBTENSOR_DIMENSIONS_H
#define LIBTENSOR_DIMENSIONS_H

#include <iostream>
#include "../defs.h"
#include "../exception.h"
#include "index.h"
#include "index_range.h"
#include "permutation.h"

namespace libtensor {

template<size_t N> class dimensions;

template<size_t N>
std::ostream &operator<<(std::ostream &os, const dimensions<N> &dims);


/**	\brief Tensor %dimensions
	\tparam N Tensor order.

	Stores the number of %tensor elements along each dimension.

	\ingroup libtensor_core
**/
template<size_t N>
class dimensions {
	friend std::ostream &operator<< <N>(std::ostream &os,
		const dimensions<N> &dims);

private:
	index<N> m_dims; //!< Tensor %dimensions
	index<N> m_incs; //!< Index increments
	size_t m_size; //!< Total size

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates a copy of another dimensions object
		\param d Another dimensions object.
	 **/
	dimensions(const dimensions<N> &d);

	/**	\brief Convers a range of indexes to the dimensions object
		\param r Index range.
	 **/
	dimensions(const index_range<N> &r);

	//@}


	//!	\name Dimensions manipulations, comparison, etc.
	//@{

	/**	\brief Returns the total number of elements
	 **/
	size_t get_size() const;

	/**	\brief Returns the number of elements along a given dimension
	 **/
	size_t get_dim(size_t i) const throw(exception);

	/**	\brief Returns the linear increment along a given dimension
	 **/
	size_t get_increment(size_t i) const throw(exception);

	/**	\brief Returns true if an %index is within the %dimensions
	 **/
	bool contains(const index<N> &idx) const;

	/**	\brief Returns true if two %dimensions objects are equal
	 **/
	bool equals(const dimensions<N> &other) const;

	/**	\brief Permutes the %dimensions
		\return The reference to the current %dimensions object
	 **/
	dimensions<N> &permute(const permutation<N> &p) throw(exception);

	//@}


	//!	\name Overloaded operators
	//@{

	/**	\brief Returns the number of elements along a given dimension
	 **/
	size_t operator[](size_t i) const throw(exception);

	/**	\brief Returns true if two %dimensions objects are equal
	 **/
	bool operator==(const dimensions<N> &other) const;

	/**	\brief Returns true if two %dimensions objects are different
	 **/
	bool operator!=(const dimensions<N> &other) const;

	//@}


	//!	\name Index manipulations
	//@{

	/**	\brief Increments an %index within the %dimensions
		\param i Index.
		\return True on success and false if the index cannot be
			incremented (points to the last element or out of
			bounds).
		\throw exception If the index is incompatible with the
			dimensions object.
	 **/
	bool inc_index(index<N> &idx) const throw(exception);

	/**	\brief Returns the absolute %index within the %dimensions
			(last %index is the fastest)
	 **/
	size_t abs_index(const index<N> &idx) const throw(exception);

	/**	\brief Converts an absolute %index back to a normal %index
	 **/
	void abs_index(const size_t abs, index<N> &idx) const throw(exception);

	//@}

private:
	/**	\brief Updates the linear increments for each dimension
	 **/
	void update_increments();

};

template<size_t N>
inline dimensions<N>::dimensions(const dimensions<N> &d)
	: m_dims(d.m_dims), m_incs(d.m_incs), m_size(d.m_size) {
}

template<size_t N>
inline dimensions<N>::dimensions(const index_range<N> &r)
	: m_dims(r.get_end()) {
	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++) {
		m_dims[i] -= r.get_begin()[i];
		m_dims[i]++;
	}
	update_increments();
}

template<size_t N>
inline size_t dimensions<N>::get_size() const {
	return m_size;
}

template<size_t N>
inline size_t dimensions<N>::get_dim(size_t i) const throw(exception) {
	return m_dims[i];
}

template<size_t N>
inline size_t dimensions<N>::get_increment(size_t i) const throw(exception) {
	return m_incs[i];
}

template<size_t N>
inline bool dimensions<N>::contains(const index<N> &idx) const {
	#pragma unroll(N)
	for(register size_t i = 0; i < N; i++) {
		if(idx[i] >= m_dims[i]) return false;
	}
	return true;
}

template<size_t N>
inline bool dimensions<N>::equals(const dimensions<N> &other) const {
	return m_dims.equals(other.m_dims);
}

template<size_t N>
inline dimensions<N> &dimensions<N>::permute(const permutation<N> &p)
	throw(exception) {
	m_dims.permute(p);
	update_increments();
	return *this;
}

template<size_t N>
bool dimensions<N>::inc_index(index<N> &idx) const throw(exception) {
	if(!contains(idx)) return false;
	size_t n = N - 1;
	bool done = false, ok = false;
	do {
		if(idx[n] < m_dims[n]-1) {
			idx[n]++;
			for(size_t i=n+1; i<N; i++) idx[i]=0;
			done = true; ok = true;
		} else {
			if(n == 0) done = true;
			else n--;
		}
	} while(!done);
	return ok;
}

template<size_t N>
size_t dimensions<N>::abs_index(const index<N> &idx) const throw(exception) {
	size_t abs = 0;
	for(register size_t i=0; i<N; i++) {
		if(idx[i] < m_dims[i]) {
			abs += m_incs[i]*idx[i];
		} else {
			throw_exc("dimensions<N>", "abs_index(const index<N>&)",
				"Index out of range");
		}
	}
	return abs;
}

template<size_t N>
void dimensions<N>::abs_index(const size_t abs, index<N> &idx) const
	throw(exception) {
	size_t a = abs;
	register size_t imax = N-1;
	for(register size_t i=0; i<imax; i++) {
		idx[i] = a/m_incs[i];
		a %= m_incs[i];
	}
	idx[N-1] = a;
}

template<size_t N>
void dimensions<N>::update_increments() {
	register size_t sz = 1;
	register size_t i = N;
	while(i != 0) {
		i--;
		m_incs[i] = sz; sz *= m_dims[i];
	}
	m_size = sz;
}

template<size_t N>
inline size_t dimensions<N>::operator[](size_t i) const throw(exception) {
	return get_dim(i);
}

template<size_t N>
inline bool dimensions<N>::operator==(const dimensions<N> &other) const {
	return equals(other);
}

template<size_t N>
inline bool dimensions<N>::operator!=(const dimensions<N> &other) const {
	return !equals(other);
}

template<size_t N>
std::ostream &operator<<(std::ostream &os, const dimensions<N> &dims) {
	os << dims.m_dims;
	return os;
}


} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_H

