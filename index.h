#ifndef LIBTENSOR_INDEX_H
#define LIBTENSOR_INDEX_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Index of a single %tensor element

	A correct %index must have the same order as the %tensor, and none of
	the %index elements must be out of the range of the %tensor
	%dimensions.

	The elements of an %index can be permuted by a %permutation. Since
	there can be multiple implementations of permutations, the method
	permute() is a template. For more info \ref permutations.

	\ingroup libtensor
**/
template<size_t N>
class index {
private:
	size_t m_idx[N]; //!< Tensor %index

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the %index of the first element of a %tensor
			 with a given order
	**/
	index();

	/**	\brief Copies the %index from another instance
		\param idx Another %index.
	**/
	index(const index<N> &idx);

	//@}

	/**	\brief Checks if two indices are equal

		Returns true if the indices are equal, false otherwise.
	**/
	bool equals(const index<N> &idx) const;

	/**	\brief Compares two indexes
		\return true if this %index is smaller than the other one,
			false otherwise.
	**/
	bool less(const index<N> &idx) const;

	/**	\brief Lvalue individual element accessor

		Returns an individual %index element at the position \e i as an
		lvalue.

		\param i Element position.
		\return Reference to the element.
		\throw exception If the position is out of range.
	**/
	size_t &operator[](const size_t i) throw(exception);

	/**	\brief Rvalue individual element accessor

		Returns an individual %index element at the position \e i as an
		rvalue.

		\param i Element position.
		\return Element value.
		\throw exception If the position is out of range.
	**/
	size_t operator[](const size_t i) const throw(exception);

	/**	\brief Permutes the elements of the %index
		\param p Permutation.
		\return Reference to this %index.
		\throw exception If the %index and the %permutation are
			incompatible.
	**/
	index<N> &permute(const permutation<N> &p) throw(exception);

};

/**	\brief Specialized %index for zeroth-order tensors

	\ingroup libtensor
**/
template<>
class index<0> {
};

template<size_t N>
inline index<N>::index() {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[i] = 0;
}

template<size_t N>
inline index<N>::index(const index<N> &idx) {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[i] = idx.m_idx[i];
}

template<size_t N>
inline bool index<N>::equals(const index<N> &idx) const {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++)
		if(m_idx[i] != idx.m_idx[i]) return false;
	return true;
}

template<size_t N>
inline bool index<N>::less(const index<N> &idx) const {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) {
		if(m_idx[i] < idx.m_idx[i]) return true;
		if(m_idx[i] > idx.m_idx[i]) return false;
	}
	return false;
}

template<size_t N>
inline size_t &index<N>::operator[](const size_t i) throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(i >= N) {
		throw_exc("index<N>", "operator[](const size_t)",
			"Index out of range");
	}
#endif // LIBTENSOR_DEBUG
	return m_idx[i];
}

template<size_t N>
inline size_t index<N>::operator[](const size_t i) const throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(i >= N) {
		throw_exc("index<N>", "operator[](const size_t) const",
			"Index out of range");
	}
#endif // LIBTENSOR_DEBUG
	return m_idx[i];
}

template<size_t N>
inline index<N> &index<N>::permute(const permutation<N> &p) throw(exception) {
	p.apply(N, m_idx);
	return *this;
}

} // namespace libtensor

#endif // LIBTENSOR_INDEX_H

