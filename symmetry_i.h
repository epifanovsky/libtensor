#ifndef LIBTENSOR_SYMMETRY_I_H
#define LIBTENSOR_SYMMETRY_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Permutational symmetry of blocks

	\ingroup libtensor
**/
class symmetry_i {
public:
	/**	\brief Returns the index of the unique block
	**/
	virtual const index &get_unique(const index &i) const = 0;

	virtual const permutation &get_perm(const index &i) const = 0;

	virtual double get_coeff(const index &i) const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_I_H

