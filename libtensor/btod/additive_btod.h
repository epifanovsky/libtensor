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
	typedef typename schedule_t::schedule_node_t schedule_node_t;
	typedef typename schedule_t::tier3_list_t tier3_list_t;
	typedef typename schedule_t::tier4_list_t tier4_list_t;

private:
	void process_tier1(bool zeroa, tensor_i<N, double> &blk,
		const index<N> &ia, double c);
	void process_tier2(block_tensor_ctrl<N, double> &ctrl,
		const index<N> &ia, const index<N> &ib,
		const transf<N, double> &tr, double c);
	void process_tier3(block_tensor_ctrl<N, double> &ctrl,
		const dimensions<N> &bidims, tensor_i<N, double> &auxblk,
		const index<N> &ia, const tier3_list_t &tier3);
	void process_tier4(block_tensor_ctrl<N, double> &ctrl,
		const dimensions<N> &bidims, tensor_i<N, double> &auxblk,
		const tier4_list_t &tier4);

};


template<size_t N>
inline void additive_btod<N>::compute_block(additive_btod<N> &op,
	tensor_i<N, double> &blk, const index<N> &i,
	const transf<N, double> &tr, double c) {

	op.compute_block(blk, i, tr, c);
}


template<size_t N>
void additive_btod<N>::process_tier1(bool zeroa, tensor_i<N, double> &blk,
	const index<N> &ia, double c) {

	//~ std::cout << "T1 " << ia << (zeroa ? " Z" : "") << std::endl;
	transf<N, double> tr0;
	if(zeroa) tod_set<N>().perform(blk);
	compute_block(blk, ia, tr0, c);
}


template<size_t N>
void additive_btod<N>::process_tier2(block_tensor_ctrl<N, double> &ctrl,
	const index<N> &ia, const index<N> &ib, const transf<N, double> &tr,
	double c) {

	bool zerob = ctrl.req_is_zero_block(ib);

	//~ std::cout << "T2 " << ib << "<-" << ia << (zerob ? " Z" : "") << std::endl;
	tensor_i<N, double> &blk = ctrl.req_block(ib);
	if(zerob) tod_set<N>().perform(blk);
	compute_block(blk, ia, tr, c);
	ctrl.ret_block(ib);
}


template<size_t N>
void additive_btod<N>::process_tier3(block_tensor_ctrl<N, double> &ctrl,
	const dimensions<N> &bidims, tensor_i<N, double> &auxblk,
	const index<N> &ia, const tier3_list_t &tier3) {

	bool zeroa = ctrl.req_is_zero_block(ia);

	tensor_i<N, double> &blk = ctrl.req_block(ia);
	if(zeroa) tod_copy<N>(auxblk).perform(blk);
	else tod_copy<N>(auxblk).perform(blk, 1.0);
	ctrl.ret_block(ia);

	for(typename tier3_list_t::const_iterator j = tier3.begin(); j != tier3.end(); j++) {

		abs_index<N> aib(j->cib, bidims);
		bool zerob = ctrl.req_is_zero_block(aib.get_index());
		tensor_i<N, double> &blk = ctrl.req_block(aib.get_index());
		if(zerob) tod_copy<N>(auxblk, j->tra.get_perm(), j->tra.get_coeff()).perform(blk);
		else tod_copy<N>(auxblk, j->tra.get_perm(), j->tra.get_coeff()).perform(blk, 1.0);
		ctrl.ret_block(aib.get_index());
	}
}


template<size_t N>
void additive_btod<N>::process_tier4(block_tensor_ctrl<N, double> &ctrl,
	const dimensions<N> &bidims, tensor_i<N, double> &auxblk,
	const tier4_list_t &tier4) {

	for(typename tier4_list_t::const_iterator j = tier4.begin(); j != tier4.end(); j++) {

		abs_index<N> aib(j->cib, bidims);
		abs_index<N> aic(j->cic, bidims);

		//~ std::cout << "T4 " << aic.get_index() << "<-" << aib.get_index() << std::endl;
		bool zerob = ctrl.req_is_zero_block(aib.get_index());
		if(zerob) {
			tensor_i<N, double> &blkc = ctrl.req_block(aic.get_index());
			tod_copy<N>(auxblk, j->tra.get_perm(), j->tra.get_coeff()).perform(blkc);
			ctrl.ret_block(aic.get_index());
		} else {
			tensor_i<N, double> &blkb = ctrl.req_block(aib.get_index());
			tensor_i<N, double> &blkc = ctrl.req_block(aic.get_index());
			tod_add<N> op(auxblk, j->tra.get_perm(), j->tra.get_coeff());
			op.add_op(blkb, j->trb.get_perm(), j->trb.get_coeff());
			op.perform(blkc);
			ctrl.ret_block(aic.get_index());
			ctrl.ret_block(aib.get_index());
		}
	}
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

	for(typename schedule_t::iterator i = sch.begin();
		i != sch.end(); i++) {

		const schedule_node_t &n = sch.get_node(i);

		abs_index<N> aia(n.cia, bidims);
		bool zeroa = ctrl.req_is_zero_block(aia.get_index());

		if(n.tier2 == 0) {

			if(n.tier3 == 0 && n.tier4 == 0) {

				tensor_i<N, double> &blk = ctrl.req_block(aia.get_index());
				process_tier1(zeroa, blk, aia.get_index(), c);
				ctrl.ret_block(aia.get_index());

			} else {

				tensor_i<N, double> &auxblk = ctrl.req_aux_block(aia.get_index());
				process_tier1(true, auxblk, aia.get_index(), c);

				if(n.tier4 != 0) {
					process_tier4(ctrl, bidims, auxblk, *n.tier4);
				}

				if(n.tier3 != 0) {

					process_tier3(ctrl, bidims, auxblk, aia.get_index(), *n.tier3);
				}
				ctrl.ret_aux_block(aia.get_index());

			}

		} else {

			if(n.tier3 == 0 && n.tier4 == 0) {

				abs_index<N> aib(n.tier2->cib, bidims);
				process_tier2(ctrl, aia.get_index(), aib.get_index(), n.tier2->trb, c);

			}
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_H
