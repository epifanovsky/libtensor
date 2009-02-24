#ifndef LIBTENSOR_SYMMETRY_I_H
#define LIBTENSOR_SYMMETRY_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Interface to %symmetry of blocks in a block %tensor

	Each block in a block %tensor can be either a unique block or a replica
	of a unique block with a permutation and/or a coefficient applied
	to the elements of the block. For example, a block matrix that is
	symmetric with respect to its diagonal will have identical blocks
	across the diagonal: only those above the diagonal need to be actually
	stored, the blocks below the diagonal are their transpose.
	When there is no %symmetry at all, every block is unique.

	Unique blocks must have the identity permutation and the coefficient
	1.0.
	
	\ingroup libtensor
**/
class symmetry_i {
public:
	/**	\brief Checks if the block is unique
		\param i Block index.
	**/
	virtual bool is_unique(const index &i) const throw(exception) = 0;

	/**	\brief Returns the index of the unique block
		\param i Block index.
	**/
	virtual const index &get_unique(const index &i) const 
		throw(exception) = 0;

	/**	\brief Returns the %permutation that needs to be applied to
			the unique block to obtain the replica
		\param i Block index.
	**/
	virtual const permutation &get_perm(const index &i) const
		throw(exception) = 0;

	/**	\brief Returns the coefficient that needs to be applied to
			the elements of the unique block to obtain the
			replica
		\param i Block index.
	**/
	virtual double get_coeff(const index &i) const throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_I_H

