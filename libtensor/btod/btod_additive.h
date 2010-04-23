#ifndef LIBTENSOR_BTOD_ADDITIVE_H
#define LIBTENSOR_BTOD_ADDITIVE_H

#include "../defs.h"
#include "../core/direct_block_tensor_operation.h"
#include "../symmetry/so_copy.h"
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
class btod_additive : public direct_block_tensor_operation<N, double> {
public:
	using direct_block_tensor_operation<N, double>::get_bis;
	using direct_block_tensor_operation<N, double>::get_symmetry;
	using direct_block_tensor_operation<N, double>::compute_block;

	virtual void perform(block_tensor_i<N, double> &bt);

	//!	\name Interface of additive operations
	//@{

	/**	\brief Returns the assignment schedule -- the preferred order
			of computing blocks
	 **/
	virtual const assignment_schedule<N, double> &get_schedule() = 0;

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


template<size_t N>
void btod_additive<N>::perform(block_tensor_i<N, double> &bt) {

	block_tensor_ctrl<N, double> ctrl(bt);
	ctrl.req_zero_all_blocks();
	so_copy<N, double>(get_symmetry()).perform(ctrl.req_symmetry());

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	const assignment_schedule<N, double> &sch = get_schedule();
	for(typename assignment_schedule<N, double>::iterator i = sch.begin();
		i != sch.end(); i++) {

		abs_index<N> ai(sch.get_abs_index(i), bidims);
		tensor_i<N, double> &blk = ctrl.req_block(ai.get_index());
		compute_block(blk, ai.get_index());
		ctrl.ret_block(ai.get_index());
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADDITIVE_H
