#ifndef LIBTENSOR_ADDITIVE_BTOD_H
#define LIBTENSOR_ADDITIVE_BTOD_H

#include "basic_btod.h"
#include "addition_schedule.h"

namespace libtensor {


/**	\brief Base class for additive block %tensor operations
	\tparam N Tensor order.

	Additive block %tensor operations are those that can add their result
	to the output block %tensor as opposed to simply replacing it. This
	class extends basic_btod<N> with two new functions: one is invoked
	to perform the block %tensor operation additively, the other does that
	for only one canonical block.

	The coefficient provided in both functions scales the result of the
	operation before adding it to the output block %tensor.

	\ingroup libtensor_btod
 **/
template<size_t N>
class additive_btod : public basic_btod<N> {
public:
	using basic_btod<N>::get_bis;
	using basic_btod<N>::get_symmetry;
	using basic_btod<N>::perform;

public:
	/**	\brief Computes the result of the operation and adds it to the
			output block %tensor
		\param bt Output block %tensor.
		\param c Scaling coefficient.
	 **/
	virtual void perform(block_tensor_i<N, double> &bt, double c);

protected:
	using basic_btod<N>::compute_block;

protected:
	/**	\brief Computes a single block of the result and adds it to
			the output %tensor
		\param blk Output %tensor.
		\param i Index of the block to compute.
		\param c Scaling coefficient.
	 **/
	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		double c) = 0;

};


template<size_t N>
void additive_btod<N>::perform(block_tensor_i<N, double> &bt, double c) {

}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_H
