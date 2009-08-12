#ifndef LIBTENSOR_BLOCK_TENSOR_I_H
#define LIBTENSOR_BLOCK_TENSOR_I_H

#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "index.h"
#include "orbit_iterator.h"
#include "symmetry.h"
#include "tensor_i.h"

namespace libtensor {

template<size_t N, typename T> class block_tensor_ctrl;

/**	\brief Block %tensor interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	Block tensors break down the whole %tensor into small blocks and
	implement %symmetry relations between them. Each individual block is
	a small %tensor itself and can be addressed using the tensor_i<N, T>
	interface.

	The block %tensor interface does not impose any restrictions on how
	a particular implementation operates. However, it defines a set of
	events which have to reacted upon. The overall mechanism is the same
	as for regular tensors (see tensor_i<N, T>): a control object (an
	instance of block_tensor_ctrl<N, T>) mediates requests to the block
	%tensor object and ensures the completeness of the interaction session.

	<b>Block %tensor events</b>

	<b>req_symmetry</b> The request for %symmetry is invoked when the client
	requires access to the %symmetry object. The handler returns a reference
	to %symmetry.

	<b>req_orbits</b> Invoked to obtain an iterator over all non-zero
	orbits (%symmetry-unique blocks) in the block %tensor. The handler
	returns an instance of orbit_iterator<N, T>.

	<b>req_block</b> Invoked to obtain a canonical block with a given
	%index.

	<b>ret_block</b> Invoked to indicate that a canonical block with a
	given %index is not required anymore.

	<b>req_zero_block</b> Invoked to inform that a canonical block has all
	its elements equal to zero.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class block_tensor_i {
	friend class block_tensor_ctrl<N, T>;

public:
	/**	\brief Returns the block %index space of the block %tensor
	 **/
	virtual const block_index_space<N> &get_bis() const = 0;

protected:
	//!	\name Event handling
	//@{

	/**	\brief Invoked when the %symmetry object is requested
		\return The %symmetry of the block %tensor
	 **/
	//virtual symmetry<N, T> &on_req_symmetry() throw(exception) = 0;

	/**	\brief Invoked when a canonical block is requested
		\param idx Block %index.
		\return Reference to the requested block.
	 **/
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception) = 0;

	/**	\brief Invoked to return a canonical block
		\param idx Block %index.
	 **/
	virtual void on_ret_block(const index<N> &idx) throw(exception) = 0;

	/**	\brief Invoked to make a canonical block zero
		\param idx Block %index.
	 **/
	virtual void on_req_zero_block(const index<N> &idx)
		throw(exception) = 0;

	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_I_H
