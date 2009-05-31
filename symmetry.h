#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <map>

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Stores %symmetry information about blocks in a block %tensor

	\ingroup libtensor
**/
template<size_t N, typename T>
class symmetry {
private:
	const symmetry_i<N, T> &m_sym; //!< Symmetry object
	dimensions<N> &m_dims; //!< Block %index dimensions

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the %symmetry object
		\param sym Symmetry.
		\param dims Dimensions to which the symmetry is applied.
	**/
	symmetry(const symmetry_i<N, T> &sym, const dimensions<N> &dims);

	//@}
};

template<size_t N, typename T>
symmetry<N, T>::symmetry(const symmetry_i<N, T> &sym,
	const dimensions<N> &dims)
	: m_sym(sym), m_dims(dims) {

}

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H

