#ifndef LIBTENSOR_BTOD_SET_ELEM_H
#define LIBTENSOR_BTOD_SET_ELEM_H

#include "../defs.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../tod/tod_set.h"
#include "../tod/tod_set_elem.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Sets a single element of a block %tensor to a value
	\tparam N Tensor order.

	The operation sets one block %tensor element specified by a block
	%index and an %index within the block. The symmetry is preserved.
	If the affected block shares an orbit with other blocks, those will
	be affected accordingly.

	Normally for clarity reasons the block %index used with this operation
	should be canonical. If it is not, the canonical block is changed using
	%symmetry rules such that the specified element of the specified block
	is given the specified value.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_set_elem {
public:
	static const char *k_clazz; //!< Class name

public:
	/**	\brief Default constructor
	 **/
	btod_set_elem() { }

	/**	\brief Performs the operation
		\param bt Block %tensor.
		\param bidx Block %index.
		\param idx Element %index within the block.
		\param d Element value.
	 **/
	void perform(block_tensor_i<N, double> &bt, const index<N> &bidx,
		const index<N> &idx, double d);

private:
	btod_set_elem(const btod_set_elem<N> &);
	const btod_set_elem<N> &operator=(const btod_set_elem<N> &);

};


template<size_t N>
const char *btod_set_elem<N>::k_clazz = "btod_set_elem<N>";


template<size_t N>
void btod_set_elem<N>::perform(block_tensor_i<N, double> &bt,
	const index<N> &bidx, const index<N> &idx, double d) {

	block_tensor_ctrl<N, double> ctrl(bt);

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	orbit<N, double> o(ctrl.req_symmetry(), bidx);
	const transf<N, double> &tr = o.get_transf(bidx);
	abs_index<N> abidx(o.get_abs_canonical_index(), bidims);

	bool zero = ctrl.req_is_zero_block(abidx.get_index());
	tensor_i<N, double> &blk = ctrl.req_block(abidx.get_index());

	if(zero) tod_set<N>().perform(blk);

	permutation<N> perm(tr.get_perm(), true);
	index<N> idx1(idx); idx1.permute(perm);
	double d1 = d / tr.get_coeff();
	tod_set_elem<N>().perform(blk, idx1, d1);

	ctrl.ret_block(abidx.get_index());
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_ELEM_H
