#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <map>

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "symmetry_i.h"

namespace libtensor {

/**	\brief Stores %symmetry information about blocks in a block %tensor

	The concept of block %symmetry described in libtensor::symmetry_i.

	\ingroup libtensor
**/
class symmetry : public symmetry_i {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Makes a deep copy of another symmetry object
		\param s Symmetry.
		\param d Dimensions to which the symmetry is applied.
	**/
	symmetry(const symmetry_i &s, const dimensions &d);

	/**	\brief Virtual destructor
	**/
	virtual ~symmetry();

	//@}

	//!	\name Implementation of symmetry_i
	//@{
	virtual const index &get_unique(const index &i) const throw(exception);
	virtual const permutation &get_perm(const index &i) const
		throw(exception);
	virtual double get_coeff(const index &i) const throw(exception);
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H

