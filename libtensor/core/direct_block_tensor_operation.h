#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

#include "../defs.h"
#include "../exception.h"
#include "block_index_space.h"
#include "block_tensor_i.h"
#include "symmetry.h"

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
	/**	\brief Returns the block %index space of the result
	 **/
	virtual const block_index_space<N> &get_bis() const = 0;

	/**	\brief Returns the symmetry of the result
	 **/
	virtual const symmetry<N, T> &get_symmetry() const = 0;

	/**	\brief Invoked to execute the operation
	 **/
	virtual void perform(block_tensor_i<N, T> &bt) throw(exception) = 0;

	/**	\brief Invoked to calculate one block
	 **/
	virtual void perform(block_tensor_i<N, T> &bt, const index<N> &i)
		throw(exception) = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_OPERATION_H

