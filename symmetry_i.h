#ifndef LIBTENSOR_SYMMETRY_I_H
#define LIBTENSOR_SYMMETRY_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"

namespace libtensor {

template<size_t N, typename T> class symmetry_operation_i;
template<size_t N, typename T> class orbit_iterator;

/**	\brief Block %tensor symmetry interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	Symmetry in block tensors is defined through a symmetry group that
	acts on block indexes of the %tensor. Given a set of all block indexes
	and a symmetry group, a group action is defined. The action yields
	orbits (equivalence classes), and the transversal of the orbits produces
	an %index set partition: each %index from the set belongs to only one
	orbit. In addition, only one %index from each orbit corresponds to a
	unique block in the %tensor. The entire orbit can be reproduced using
	the elements of the symmetry group and one member of the orbit.
	Therefore, the block %tensor only has to store one block per orbit
	(canonical block).

	This interface assumes the "black box" character of symmetry, many
	operations can be done without knowing the actual implementation.
	However, since different symmetry classes have properties that are
	specific to them, operations that go beyond iterating orbits and
	indexes must be done through a gateway provided by invoke().

	\ingroup libtensor
**/
template<size_t N, typename T>
class symmetry_i {
public:
	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_i() { }

	/**	\brief Checks if two symmetry objects belong to the same
			class
		\param other Second symmetry class.
	 **/
	virtual bool is_same(const symmetry_i<N, T> &other) const = 0;

	/**	\brief Returns the orbit iterator
		\param dims Dimensions of the block %index space.
	 **/
	virtual orbit_iterator<N, T> get_orbits(const dimensions<N> &dims) = 0;

	/**	\brief Invokes a symmetry operation using the double dispatch
			mechanism
		\param op Symmetry operation.
		\throw exception If the symmetry operation causes an %exception
	 **/
	virtual void invoke(symmetry_operation_i<N, T> &op)
		throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_I_H

