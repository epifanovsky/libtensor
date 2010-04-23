#ifndef LIBTENSOR_BTOD_ADDITIVE_H
#define LIBTENSOR_BTOD_ADDITIVE_H

#include "../defs.h"
#include "../core/direct_block_tensor_operation.h"
#include "../symmetry/so_copy.h"
#include "basic_btod.h"
#include "assignment_schedule.h"
#include "addition_schedule.h"

namespace libtensor {


/**	\brief Additive direct block %tensor operation
	\tparam N Tensor order.

	Additive block %tensor operations are those that can add their result
	to the output block %tensor as opposed to simply replacing it. This
	base class extends direct_block_tensor_operation<N, double> with two
	functions: one is invoked to compute the result of the block %tensor
	operation and add it with a coefficient, the other does that for only
	one canonical block.

	The coefficient provided in both functions scales the result of the
	operation before adding it to the output block %tensor.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_additive : public basic_btod<N> {
public:
	using basic_btod<N>::get_bis;
	using basic_btod<N>::get_symmetry;
	using basic_btod<N>::perform;

public:
	//!	\name Interface of additive operations
	//@{

	/**	\brief Invoked to execute the operation (additive)
		\param bt Output block %tensor.
		\param c Coefficient.
	 **/
	virtual void perform(block_tensor_i<N, double> &bt, double c) = 0;

	/**	\brief Invoked to calculate one block (additive)
		\param bt Output block %tensor.
		\param i Index of the block to calculate.
		\param c Coefficient.
	 **/
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &i,
		double c) = 0;

	//@}

protected:
	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		double c) = 0;


};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADDITIVE_H
