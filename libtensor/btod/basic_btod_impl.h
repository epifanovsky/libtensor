#ifndef LIBTENSOR_BASIC_BTOD_IMPL_H
#define LIBTENSOR_BASIC_BTOD_IMPL_H

namespace libtensor {


template<size_t N>
void basic_btod<N>::perform(block_tensor_i<N, double> &bt) {

	sync_on();

	block_tensor_ctrl<N, double> ctrl(bt);
	ctrl.req_sync_on();
	ctrl.req_zero_all_blocks();
	so_copy<N, double>(get_symmetry()).perform(ctrl.req_symmetry());

	std::vector<task*> tasks;
	task_batch batch;

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	const assignment_schedule<N, double> &sch = get_schedule();
	for(typename assignment_schedule<N, double>::iterator i = sch.begin();
		i != sch.end(); i++) {

		task *t = new task(*this, bt, bidims, sch, i);
		tasks.push_back(t);
		batch.push(*t);
	}

	batch.wait();
	for(typename std::vector<task*>::iterator i = tasks.begin();
		i != tasks.end(); i++) delete *i;
	tasks.clear();
	ctrl.req_sync_off();

	sync_off();
}


template<size_t N>
void basic_btod<N>::task::perform() throw(exception) {

	block_tensor_ctrl<N, double> ctrl(m_bt);
	abs_index<N> ai(m_sch.get_abs_index(m_i), m_bidims);
	tensor_i<N, double> &blk = ctrl.req_block(ai.get_index());
	m_btod.compute_block(blk, ai.get_index());
	ctrl.ret_block(ai.get_index());
}


} // namespace libtensor

#endif // LIBTENSOR_BASIC_BTOD_IMPL_H
