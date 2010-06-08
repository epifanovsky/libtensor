#ifndef LIBTENSOR_ADDITIVE_BTOD_H
#define LIBTENSOR_ADDITIVE_BTOD_H

#include <cmath>
#include "../symmetry/so_add.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_set.h"
#include "basic_btod.h"
#include "addition_schedule.h"
#include "transf_double.h"

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
	using basic_btod<N>::get_schedule;
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
		\param tr Transformation of the block.
		\param c Scaling coefficient.
	 **/
	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		const transf<N, double> &tr, double c) = 0;

	/**	\brief Invokes compute_block on another additive operation;
			allows derived classes to call other additive operations
	 **/
	void compute_block(additive_btod<N> &op, tensor_i<N, double> &blk,
		const index<N> &i, const transf<N, double> &tr, double c);

private:
	typedef addition_schedule<N, double> schedule_t;

};


template<size_t N>
inline void additive_btod<N>::compute_block(additive_btod<N> &op,
	tensor_i<N, double> &blk, const index<N> &i,
	const transf<N, double> &tr, double c) {

	op.compute_block(blk, i, tr, c);
}


template<size_t N>
void additive_btod<N>::perform(block_tensor_i<N, double> &bt, double c) {

	if(fabs(c) == 0.0) return;

	block_tensor_ctrl<N, double> ctrl(bt);
	symmetry<N, double> symcopy(bt.get_bis());
	so_copy<N, double>(ctrl.req_const_symmetry()).perform(symcopy);
	permutation<N> p0;
	so_add<N, double>(get_symmetry(), p0, symcopy, p0).
		perform(ctrl.req_symmetry());


	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	schedule_t sch(get_symmetry(), symcopy);
	sch.build(get_schedule());

	for(typename schedule_t::iterator igrp = sch.begin();
		igrp != sch.end(); igrp++) {

		const typename schedule_t::schedule_group &grp =
			sch.get_node(igrp);

//		std::cout << "grp " << &grp << " sz = " << grp.lst.size() << std::endl;

		typedef std::pair<size_t, tensor_i<N, double>*> la_pair_t;
		std::list<la_pair_t> la;

		for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
			grp.lst.begin(); inode != grp.lst.end(); inode++) {

			const typename schedule_t::schedule_node &node = *inode;

			typename std::list<la_pair_t>::iterator ila = la.begin();
			for(; ila != la.end(); ila++) {
				if(ila->first == node.cia) break;
			}
			if(ila == la.end()) {
				abs_index<N> aia(node.cia, bidims);
				tensor_i<N, double> &blka = ctrl.req_aux_block(aia.get_index());
				tod_set<N>().perform(blka);
				compute_block(blka, aia.get_index(), node.tra, c);
//				std::cout << "compute A" << aia.get_index() << "(" << node.tra.get_perm() << ", " << node.tra.get_coeff() << ") " << c << std::endl;
				la.push_back(la_pair_t(node.cia, &blka));
			}
		}

		for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
			grp.lst.begin(); inode != grp.lst.end(); inode++) {

			const typename schedule_t::schedule_node &node = *inode;
			if(node.cib == node.cic) continue;

			typename std::list<la_pair_t>::iterator ila = la.begin();
			for(; ila != la.end(); ila++) {
				if(ila->first == node.cia) break;
			}

			abs_index<N> aib(node.cib, bidims), aic(node.cic, bidims);
			bool zerob = ctrl.req_is_zero_block(aib.get_index());
			tensor_i<N, double> &blkc = ctrl.req_block(aic.get_index());
			if(zerob) {
				abs_index<N> aia(node.cia, bidims);
//				std::cout << "C" << aic.get_index() << " <- " << "A" << aia.get_index() << std::endl;
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkc);
			} else {
				abs_index<N> aia(node.cia, bidims);
//				std::cout << "C" << aic.get_index() << " <- " << "A" << aia.get_index() << " + " << "B" << aib.get_index() << std::endl;
				tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkc);
				tod_copy<N>(blkb, node.trb.get_perm(), node.trb.get_coeff()).perform(blkc, 1.0);
				ctrl.ret_block(aib.get_index());
			}
			ctrl.ret_block(aic.get_index());
		}

		for(typename std::list<typename schedule_t::schedule_node>::const_iterator inode =
			grp.lst.begin(); inode != grp.lst.end(); inode++) {

			const typename schedule_t::schedule_node &node = *inode;
			if(node.cib != node.cic) continue;

			typename std::list<la_pair_t>::iterator ila = la.begin();
			for(; ila != la.end(); ila++) {
				if(ila->first == node.cia) break;
			}

			abs_index<N> aib(node.cib, bidims);
			bool zerob = ctrl.req_is_zero_block(aib.get_index());
			tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
			if(zerob) {
				abs_index<N> aia(node.cia, bidims);
//				std::cout << "B" << aib.get_index() << " <- " << "A" << aia.get_index() << std::endl;
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkb);
			} else {
				abs_index<N> aia(node.cia, bidims);
//				std::cout << "B" << aib.get_index() << " <- " << "A" << aia.get_index() << "(" << node.tra.get_perm() << ", " << node.tra.get_coeff()<< ")" << " + " << "B" << aib.get_index() << std::endl;
				tod_copy<N>(*ila->second, node.tra.get_perm(), node.tra.get_coeff()).perform(blkb, 1.0);
			}
			ctrl.ret_block(aib.get_index());
		}

		for(typename std::list<la_pair_t>::iterator ila = la.begin();
			ila != la.end(); ila++) {

			abs_index<N> aia(ila->first, bidims);
			ctrl.ret_aux_block(aia.get_index());
		}
		la.clear();
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_H
