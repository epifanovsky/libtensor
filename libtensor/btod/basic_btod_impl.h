#ifndef LIBTENSOR_BASIC_BTOD_IMPL_H
#define LIBTENSOR_BASIC_BTOD_IMPL_H

namespace libtensor {


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

#endif // LIBTENSOR_BASIC_BTOD_IMPL_H
