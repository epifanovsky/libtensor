#ifndef LIBTENSOR_ISPACE_I_H
#define	LIBTENSOR_ISPACE_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"

namespace libtensor {

/**	\brief Index space interface
	\tparam N Index space order

	The %index space defines a range of indexes, which map %tensor
	elements. The %dimensions of the %index space is the number of %tensor
	elements along each direction.

	For example, a second-order %index space defines the indexes of a
	matrix. The number of rows and columns of the matrix is the %dimensions
	of the %index space (as well as the matrix).

	\ingroup libtensor
 **/
template<size_t N>
class ispace_i {
public:
	/** \brief Returns the dimensions of the index space
	 **/
	virtual const dimensions &dims() const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_ISPACE_I_H

