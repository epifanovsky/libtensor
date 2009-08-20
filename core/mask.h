#ifndef LIBTENSOR_MASK_H
#define LIBTENSOR_MASK_H

#include "defs.h"
#include "exception.h"
#include "sequence.h"

namespace libtensor {

template<size_t N>
class mask : public sequence<N, bool> {
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

	//!	\name Operators
	//@{

	mask<N> &operator|=(const mask<N> &other);

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
		if(sequence<N, bool>::at_nochk(i) !=
			msk.sequence<N, bool>::at_nochk(i)) return false;
	return true;
}


template<size_t N>
mask<N> &mask<N>::operator|=(const mask<N> &other) {

	for(register size_t i = 0; i < N; i++) {
		sequence<N, bool>::at_nochk(i) =
			sequence<N, bool>::at_nochk(i) ||
				other.sequence<N, bool>::at_nochk(i);
	}
	return *this;
}


} // namespace libtensor

#endif // LIBTENSOR_MASK_H
