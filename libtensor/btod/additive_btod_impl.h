#ifndef LIBTENSOR_ADDITIVE_BTOD_IMPL_H
#define LIBTENSOR_ADDITIVE_BTOD_IMPL_H

#include "../symmetry/so_copy.h"

namespace libtensor {

template<size_t N>
void additive_btod<N>::compute_block(additive_btod<N> &op,
	tensor_i<N, double> &blk, const index<N> &i,
	const transf<N, double> &tr, double c) {

	op.compute_block(blk, i, tr, c);
}


template<size_t N>
void additive_btod<N>::perform(block_tensor_i<N, double> &bt, double c) {

	if(fabs(c) == 0.0) return;

	sync_on();

	block_tensor_ctrl<N, double> ctrl(bt);
	ctrl.req_sync_on();
	symmetry<N, double> symcopy(bt.get_bis());
	so_copy<N, double>(ctrl.req_const_symmetry()).perform(symcopy);
	permutation<N> p0;
	so_add<N, double>(get_symmetry(), p0, symcopy, p0).
		perform(ctrl.req_symmetry());


	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	schedule_t sch(get_symmetry(), symcopy);
	sch.build(get_schedule(), ctrl);

	std::vector<task*> tasks;
	task_batch batch;

	for(typename schedule_t::iterator igrp = sch.begin();
		igrp != sch.end(); igrp++) {

		task *t = new task(*this, bt, bidims, sch, igrp, c);
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
void additive_btod<N>::task::perform() throw(exception) {

	block_tensor_ctrl<N, double> ctrl(m_bt);

	const typename schedule_t::schedule_group &grp = m_sch.get_node(m_i);

	//~ std::cout << "grp " << &grp << " sz = " << grp.lst.size() << std::endl;

	typedef std::pair<size_t, tensor_i<N, double>*> la_pair_t;
	std::list<la_pair_t> la;

	for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
		grp.lst.begin(); inode != grp.lst.end(); inode++) {

		const typename schedule_t::schedule_node &node = *inode;

		if(node.zeroa) continue;

		typename std::list<la_pair_t>::iterator ila = la.begin();
		for(; ila != la.end(); ila++) {
			if(ila->first == node.cia) break;
		}
		if(ila == la.end()) {
			abs_index<N> aia(node.cia, m_bidims);
			tensor_i<N, double> &blka = ctrl.req_aux_block(aia.get_index());
			tod_set<N>().perform(blka);
			m_btod.compute_block(blka, aia.get_index(), node.tra, m_c);
			//~ std::cout << "compute A" << aia.get_index() << "(" << node.tra.get_perm() << ", " << node.tra.get_coeff() << ") " << c << std::endl;
			la.push_back(la_pair_t(node.cia, &blka));
		}
	}

	for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
		grp.lst.begin(); inode != grp.lst.end(); inode++) {

		const typename schedule_t::schedule_node &node = *inode;
		if(node.cib == node.cic) continue;

		if(node.zeroa) {
			abs_index<N> aib(node.cib, m_bidims), aic(node.cic, m_bidims);
			bool zerob = ctrl.req_is_zero_block(aib.get_index());
			tensor_i<N, double> &blkc = ctrl.req_block(aic.get_index());
			if(zerob) {
				// this should actually never happen, but just in case
				//~ std::cout << "C" << aic.get_index() << " <- 0" << std::endl;
				tod_set<N>().perform(blkc);
			} else {
				//~ std::cout << "C" << aic.get_index() << " <- " << "B" << aib.get_index() << "(" << node.trb.get_perm() << ", " << node.trb.get_coeff() << ")" << std::endl;
				tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
				tod_copy<N>(blkb, node.trb.get_perm(), node.trb.get_coeff()).perform(blkc);
				ctrl.ret_block(aib.get_index());
			}
			ctrl.ret_block(aic.get_index());
		} else {
			typename std::list<la_pair_t>::iterator ila = la.begin();
			for(; ila != la.end(); ila++) {
				if(ila->first == node.cia) break;
			}

			abs_index<N> aib(node.cib, m_bidims), aic(node.cic, m_bidims);
			bool zerob = ctrl.req_is_zero_block(aib.get_index());
			tensor_i<N, double> &blkc = ctrl.req_block(aic.get_index());
			if(zerob) {
				abs_index<N> aia(node.cia, m_bidims);
				//~ std::cout << "C" << aic.get_index() << " <- " << "A" << aia.get_index() << "(" << node.tra.get_perm() << ", " << node.tra.get_coeff() << ")" << std::endl;
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkc);
			} else {
				abs_index<N> aia(node.cia, m_bidims);
				//~ std::cout << "C" << aic.get_index() << " <- " << "A" << aia.get_index() << " + " << "B" << aib.get_index() << std::endl;
				tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkc);
				tod_copy<N>(blkb, node.trb.get_perm(), node.trb.get_coeff()).perform(blkc, 1.0);
				ctrl.ret_block(aib.get_index());
			}
			ctrl.ret_block(aic.get_index());
		}
	}

	for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
		grp.lst.begin(); inode != grp.lst.end(); inode++) {

		const typename schedule_t::schedule_node &node = *inode;
		if(node.cib != node.cic) continue;
		if(node.zeroa) continue;

		typename std::list<la_pair_t>::iterator ila = la.begin();
		for(; ila != la.end(); ila++) {
			if(ila->first == node.cia) break;
		}

		abs_index<N> aib(node.cib, m_bidims);
		bool zerob = ctrl.req_is_zero_block(aib.get_index());
		tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
		if(zerob) {
			abs_index<N> aia(node.cia, m_bidims);
			//~ std::cout << "B" << aib.get_index() << " <- " << "A" << aia.get_index() << std::endl;
			tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkb);
		} else {
			abs_index<N> aia(node.cia, m_bidims);
			//~ std::cout << "B" << aib.get_index() << " <- " << "A" << aia.get_index() << "(" << node.tra.get_perm() << ", " << node.tra.get_coeff()<< ")" << " + " << "B" << aib.get_index() << std::endl;
			tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkb, 1.0);
		}
		ctrl.ret_block(aib.get_index());
	}

	for(typename std::list<la_pair_t>::iterator ila = la.begin();
		ila != la.end(); ila++) {

		abs_index<N> aia(ila->first, m_bidims);
		ctrl.ret_aux_block(aia.get_index());
	}
	la.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_IMPL_H
