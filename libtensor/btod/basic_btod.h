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
	using direct_block_tensor_operation<N, double>::get_schedule;

public:
	/**	\brief Computes the result of the operation into an output
			block %tensor
		\param bt Output block %tensor.
	 **/
	virtual void perform(block_tensor_i<N, double> &bt);

protected:
	using direct_block_tensor_operation<N, double>::compute_block;

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "basic_btod_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BASIC_BTOD_H
