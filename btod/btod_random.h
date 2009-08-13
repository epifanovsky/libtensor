#ifndef LIBTENSOR_BTOD_RANDOM_H
#define LIBTENSOR_BTOD_RANDOM_H

#include <cstdlib>
#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "block_tensor_ctrl.h"
#include "tensor_i.h"
#include "tensor_ctrl.h"

namespace libtensor {


/**	\brief Fills a block %tensor with random data without affecting its
		%symmetry

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_random {
public:
	void perform(block_tensor_i<N, double> &bt) throw(exception);
};


template<size_t N>
void btod_random<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	block_tensor_ctrl<N, double> ctrl(bt);

	size_t norbits = ctrl.req_num_orbits();
	for(size_t i = 0; i < norbits; i++) {
		orbit<N, double> orb = ctrl.req_orbit(i);
		index<N> blkidx;
		bidims.abs_index(orb.get_abs_index(), blkidx);
		tensor_i<N, double> &blk = ctrl.req_block(blkidx);
		tensor_ctrl<N, double> blk_ctrl(blk);
		double *ptr = blk_ctrl.req_dataptr();
		size_t sz = blk.get_dims().get_size();
		for(register size_t i = 0; i < sz; i++) ptr[i] = drand48();
		blk_ctrl.ret_dataptr(ptr);
		ctrl.ret_block(blkidx);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_H
