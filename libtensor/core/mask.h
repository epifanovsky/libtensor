#ifndef LIBTENSOR_MASK_H
#define LIBTENSOR_MASK_H

#include "../defs.h"
#include "../exception.h"
#include "permutation.h"
#include "sequence.h"

namespace libtensor {

template<size_t N> class mask;
template<size_t N>
std::ostream &operator<<(std::ostream &os, const mask<N> &m);

template<size_t N>
class mask : public sequence<N, bool> {
	friend std::ostream &operator<< <N>(std::ostream &os, const mask<N> &m);

public:
	//!	\name Construction and destruction
	//@{

	mask();
	mask(const mask<N> &msk);

	//@}


	//!	\name Comparison
	//@{

	/**	\brief Checks if two masks are equal
	 **/
	bool equals(const mask<N> &msk) const;

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Permutes the mask
	 **/
	mask<N> &permute(const permutation<N> &perm);

	//@}

	//!	\name Operators
	//@{

	mask<N> &operator|=(const mask<N> &other);
	mask<N> operator|(const mask<N> &other) const;
	mask<N> &operator&=(const mask<N> &other);
	mask<N> operator&(const mask<N> &other) const;

	//@}
};


template<size_t N>
mask<N>::mask() : sequence<N, bool>(false) {

}


template<size_t N>
mask<N>::mask(const mask<N> &msk) : sequence<N, bool>(msk) {

}


template<size_t N>
bool mask<N>::equals(const mask<N> &msk) const {

	for(register size_t i = 0; i < N; i++)
		if(sequence<N, bool>::at_nothrow(i) !=
			msk.sequence<N, bool>::at_nothrow(i)) return false;
	return true;
}


template<size_t N>
mask<N> &mask<N>::permute(const permutation<N> &perm) {

	perm.apply(*this);
	return *this;
}


template<size_t N>
mask<N> &mask<N>::operator|=(const mask<N> &other) {

	for(register size_t i = 0; i < N; i++) {
		sequence<N, bool>::at_nothrow(i) =
			sequence<N, bool>::at_nothrow(i) ||
				other.sequence<N, bool>::at_nothrow(i);
	}
	return *this;
}


template<size_t N>
mask<N> mask<N>::operator|(const mask<N> &other) const {

	mask<N> m;
	for(register size_t i = 0; i < N; i++) {
		m.sequence<N, bool>::at_nothrow(i) =
			sequence<N, bool>::at_nothrow(i) ||
			other.sequence<N, bool>::at_nothrow(i);
	}
	return m;
}


template<size_t N>
mask<N> &mask<N>::operator&=(const mask<N> &other) {

	for(register size_t i = 0; i < N; i++) {
		sequence<N, bool>::at_nothrow(i) =
			sequence<N, bool>::at_nothrow(i) &&
				other.sequence<N, bool>::at_nothrow(i);
	}
	return *this;
}


template<size_t N>
mask<N> mask<N>::operator&(const mask<N> &other) const {

	mask<N> m;
	for(register size_t i = 0; i < N; i++) {
		m.sequence<N, bool>::at_nothrow(i) =
			sequence<N, bool>::at_nothrow(i) &&
			other.sequence<N, bool>::at_nothrow(i);
	}
	return m;
}


/**	\brief Prints out the mask to an output stream

	\ingroup libtensor
**/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const mask<N> &m) {
	os << "[";
	for(size_t j = 0; j < N; j++)
		os << m.sequence<N, bool>::at_nothrow(j) ? '1' : '0';
	os << "]";
	return os;
}


} // namespace libtensor

#endif // LIBTENSOR_MASK_H
