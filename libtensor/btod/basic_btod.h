#ifndef LIBTENSOR_BASIC_BTOD_H
#define LIBTENSOR_BASIC_BTOD_H

#include "../core/abs_index.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/direct_block_tensor_operation.h"
#include "../symmetry/so_copy.h"
#include "assignment_schedule.h"

namespace libtensor {


/**	\brief Basic functionality of block %tensor operations
	\tparam N Tensor order.

	Extends the direct_block_tensor_operation interface. Implements
	a method to compute the result in its entirety using the assignment
	schedule (preferred order of block computation) provided by derived
	block %tensor operations.

	Derived classes shall implement get_schedule().

	\sa assignment_schedule, direct_block_tensor_operation

	\ingroup libtensor_btod
 **/
template<size_t N>
class basic_btod : public direct_block_tensor_operation<N, double> {
public:
	using direct_block_tensor_operation<N, double>::get_bis;
	using direct_block_tensor_operation<N, double>::get_symmetry;

public:
	/**	\brief Returns the assignment schedule -- the preferred order
			of computing blocks
	 **/
	virtual const assignment_schedule<N, double> &get_schedule() const = 0;

public:
	/**	\brief Computes the result of the operation into an output
			block %tensor
		\param bt Output block %tensor.
	 **/
	virtual void perform(block_tensor_i<N, double> &bt);

protected:
	using direct_block_tensor_operation<N, double>::compute_block;

};


template<size_t N>
void basic_btod<N>::perform(block_tensor_i<N, double> &bt) {

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

#endif // LIBTENSOR_BASIC_BTOD_H
