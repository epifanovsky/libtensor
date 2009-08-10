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
};


template<size_t N>
mask<N>::mask() : sequence<N, bool>(false) {

}


template<size_t N>
mask<N>::mask(const mask<N> &msk) : sequence<N, bool>(msk) {

}


} // namespace libtensor

#endif // LIBTENSOR_MASK_H
