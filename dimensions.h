#ifndef LIBTENSOR_DIMENSIONS_H
#define LIBTENSOR_DIMENSIONS_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "index_range.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Contains %tensor %dimensions

	Stores the number of %tensor elements along each dimension.

	\ingroup libtensor
**/
template<size_t N>
class dimensions {
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
		\param r Index range
	**/
	dimensions(const index_range<N> &r);

	//@}

	/**	\brief Returns the linear increment along a given dimension
	**/
	size_t get_increment(const size_t i) const throw(exception);

	/**	\brief Returns the number of elements along a given dimension
	**/
	size_t operator[](const size_t i) const throw(exception);

	/**	\brief Returns the total number of elements
	**/
	size_t get_size() const;

	/**	\brief Permutes the dimensions
	**/
	dimensions<N> &permute(const permutation<N> &p) throw(exception);

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
inline dimensions<N>::dimensions(const dimensions<N> &d) :
	m_dims(d.m_dims), m_incs(d.m_incs), m_size(d.m_size) {
}

template<size_t N>
inline dimensions<N>::dimensions(const index_range<N> &r) :
	m_dims(r.get_end()) {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) {
		m_dims[i] -= r.get_begin()[i];
		m_dims[i] ++;
	}
	update_increments();
}

template<size_t N>
inline size_t dimensions<N>::get_increment(const size_t i) const
	throw(exception) {
	return m_incs[i];
}

template<size_t N>
inline size_t dimensions<N>::operator[](const size_t i) const
	throw(exception) {
	return m_dims[i];
}

template<size_t N>
inline size_t dimensions<N>::get_size() const {
	return m_size;
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
	if(m_dims.less(idx) || m_dims.equals(idx)) return false;
	size_t n = N-1;
	bool done = false;
	while(!done && n!=0) {
		if(idx[n] < m_dims[n]-1) {
			idx[n]++;
			for(size_t i=n+1; i<N; i++) idx[i]=0;
			done = true;
		} else {
			n--;
		}
	}
	return done;
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
inline void dimensions<N>::update_increments() {
	register size_t sz = 1;
	register size_t i = N;
	do {
		i--;
		m_incs[i] = sz; sz *= m_dims[i];
	} while(i != 0);
	m_size = sz;
}

//!	\name Comparisons 
//@{
/**	\brief Compare for equality of two dimensions objects 

	\return Return true if each of the N dimensions of the two dimensions objects are equal
**/
template<size_t N>
inline bool operator==( const dimensions<N> &da, const dimensions<N> &db ) {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) {
		if(da[i]!=db[i]) return false;
	}
	return true;
}

/**	\brief Compare for inequality of two dimensions objects

	\return Return the opposite of operator== 
**/
template<size_t N>
inline bool operator!=( const dimensions<N> &da, const dimensions<N> &db ) {
	return !(da==db);
}
//@}

} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_H

