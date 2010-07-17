#ifndef LIBTENSOR_BASIC_BTOD_H
#define LIBTENSOR_BASIC_BTOD_H

#include "../core/abs_index.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/direct_block_tensor_operation.h"
#include "../symmetry/so_copy.h"
#include "../mp/task_batch.h"
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
	using direct_block_tensor_operation<N, double>::sync_on;
	using direct_block_tensor_operation<N, double>::sync_off;

private:
	class task : public task_i {
	private:
		basic_btod<N> &m_btod;
		block_tensor_i<N, double> &m_bt;
		const dimensions<N> &m_bidims;
		const assignment_schedule<N, double> &m_sch;
		typename assignment_schedule<N, double>::iterator m_i;

	public:
		task(basic_btod<N> &btod, block_tensor_i<N, double> &bt,
			const dimensions<N> &bidims,
			const assignment_schedule<N, double> &sch,
			typename assignment_schedule<N, double>::iterator i) :
			m_btod(btod), m_bt(bt), m_bidims(bidims), m_sch(sch),
			m_i(i) { }
		virtual ~task() { }
		virtual void perform() throw(exception);
	};

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
