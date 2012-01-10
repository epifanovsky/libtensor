#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

#include "../defs.h"
#include "../exception.h"
#include "block_index_space.h"
#include "block_tensor_i.h"
#include "symmetry.h"
#include "../btod/assignment_schedule.h"
#include "../mp/cpu_pool.h"

namespace libtensor {


/**	\brief Underlying operation for direct block tensors

	Block %tensor operations that serve as underlying operations for
	direct block tensors take an arbitrary number of arguments, but result
	in one block %tensor.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class direct_block_tensor_operation {
public:
	virtual ~direct_block_tensor_operation() { }

	/**	\brief Returns the block %index space of the result
	 **/
	virtual const block_index_space<N> &get_bis() const = 0;

	/**	\brief Returns the symmetry of the result
	 **/
	virtual const symmetry<N, T> &get_symmetry() const = 0;

	/**	\brief Invoked to execute the operation
	 **/
	virtual void perform(block_tensor_i<N, T> &bt) = 0;

	/**	\brief Returns the assignment schedule -- the preferred order
			of computing blocks
	 **/
	virtual const assignment_schedule<N, T> &get_schedule() const = 0;

	/**	\brief Computes a single block of the result
	 **/
	virtual void compute_block(dense_tensor_i<N, double> &blk,
		const index<N> &i, cpu_pool &cpus) = 0;

	/**	\brief Enables the synchronization of arguments
	 **/
	virtual void sync_on() = 0;

	/**	\brief Disables the synchronization of arguments
	 **/
	virtual void sync_off() = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

