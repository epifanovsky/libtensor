#ifndef LIBTENSOR_SYMMETRY_I_H
#define LIBTENSOR_SYMMETRY_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "orbit_iterator.h"
#include "symmetry_target_i.h"

namespace libtensor {

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

	<b>Basic reduction and expansion of symmetry</b>

	All realizations of the symmetry interface shall implement methods to
	completely eliminate (disable_symmetry()) and reinstate symmetry
	(enable_symmetry()). The former makes every block unique, and the
	latter enables all symmetry elements available. Other grades are to be
	accessible through specific interfaces to symmetry classes.

	<b>Further symmetry manipulations</b>

	This interface assumes the "black box" character of symmetry, many
	operations can be done without knowing the actual implementation.
	However, since different symmetry classes have properties that are
	specific to them, operations that go beyond iterating orbits and
	indexes must be done through a gateway provided by dispatch().

	<b>Dispatch mechanism</b>

	This interface defines, but not establishes specifically, a dispatch
	mechanism that can be accessed through the dispatch() method. The
	purpose is to determine the type of a symmetry object at runtime and
	perform operations that are specific to that type. The details of the
	mechanism are to be disclosed by implementations of this interface.

	\ingroup libtensor_core
**/
template<size_t N, typename T>
class symmetry_i {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_i() { }

	//@}


	//!	\name Symmetry manipulations
	//@{

	/**	\brief Disables all symmetry elements such that every block
			becomes unique
	 **/
	virtual void disable_symmetry() = 0;

	/**	\brief Enables all symmetry elements
	 **/
	virtual void enable_symmetry() = 0;

	/**	\brief Returns true if a given block is canonical
	 **/
	virtual bool is_canonical(const index<N> &idx) const = 0;

	//@}


	//!	\name Iterator handlers
	//@{

	/**	\brief Returns the orbit iterator handler
	 **/
	virtual const orbit_iterator_handler_i<N, T> &get_oi_handler()
		const = 0;

	/**	\brief Returns the block iterator handler
	 **/
	virtual const block_iterator_handler_i<N, T> &get_bi_handler()
		const = 0;

	//@}


	//!	\name Dispatching mechanism
	//@{

	/**	\brief Invokes the dispatch mechanism to determine the symmetry
			type at runtime (const)
		\param target Dispatch target.
	 **/
	virtual void dispatch(symmetry_target_i<N, T> &target) const
		throw(exception) = 0;

	/**	\brief Invokes the dispatch mechanism to determine the symmetry
			type at runtime
		\param target Dispatch target.
	 **/
	virtual void dispatch(symmetry_target_i<N, T> &target)
		throw(exception) = 0;

	//@}

};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_I_H

