#ifndef LIBTENSOR_BTOD_RANDOM_H
#define LIBTENSOR_BTOD_RANDOM_H

#include <list>
#include <map>
#include <utility>
#include <libvmm/std_allocator.h>
#include "../defs.h"
#include "../exception.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit_list.h"
#include "../core/tensor.h"
#include "../core/tensor_ctrl.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_random.h"
#include "../timings.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Fills a block %tensor with random data without affecting its
		%symmetry
	\tparam T Block %tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_random : public timings< btod_random<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	typedef timings< btod_random<N> > timings_base;
	typedef std::list< transf<N, double> > transf_list_t;
	typedef std::map<size_t, transf_list_t> transf_map_t;

public:
	/**	\brief Fills a block %tensor with random values preserving
			symmetry
		\param bt Block %tensor.
	 **/
	void perform(block_tensor_i<N, double> &bt) throw(exception);

	/**	\brief Fills one block of a block %tensor with random values
			preserving symmetry
		\param bt Block %tensor.
		\param idx Block %index in the block %tensor.
	 **/
	void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);

private:
	bool make_transf_map(const symmetry<N, double> &sym,
		const dimensions<N> &bidims, const index<N> &idx,
		const transf<N, double> &tr, transf_map_t &alltransf);

	void make_random_blk(block_tensor_ctrl<N, double> &ctrl,
		const dimensions<N> &bidims, const index<N> &idx);

private:
	btod_random<N> &operator=(const btod_random<N>&);

};


template<size_t N>
const char *btod_random<N>::k_clazz = "btod_random<N>";


template<size_t N>
void btod_random<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	timings_base::start_timer();

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	block_tensor_ctrl<N, double> ctrl(bt);

	orbit_list<N, double> orblist(ctrl.req_symmetry());
	typename orbit_list<N, double>::iterator iorbit = orblist.begin();
	for(; iorbit != orblist.end(); iorbit++) {
		make_random_blk(ctrl, bidims, orblist.get_index(iorbit));
	}

	timings_base::stop_timer();
}

template<size_t N>
void btod_random<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	timings_base::start_timer();

	dimensions<N> bidims(bt.get_bis().get_block_index_dims());
	block_tensor_ctrl<N, double> ctrl(bt);
	make_random_blk(ctrl, bidims, idx);

	timings_base::stop_timer();
}


template<size_t N>
bool btod_random<N>::make_transf_map(const symmetry<N, double> &sym,
	const dimensions<N> &bidims, const index<N> &idx,
	const transf<N, double> &tr, transf_map_t &alltransf) {

	size_t absidx = bidims.abs_index(idx);
	typename transf_map_t::iterator ilst = alltransf.find(absidx);
	if(ilst == alltransf.end()) {
		ilst = alltransf.insert(std::pair<size_t, transf_list_t>(
			absidx, transf_list_t())).first;
	}
	typename transf_list_t::iterator itr = ilst->second.begin();
	bool done = false;
	for(; itr != ilst->second.end(); itr++) {
		if(*itr == tr) {
			done = true;
			break;
		}
	}
	if(done) return true;
	ilst->second.push_back(tr);

	bool allowed = true;
	typename symmetry<N, double>::iterator ielem = sym.begin();
	for(; ielem != sym.end(); ielem++) {
		const symmetry_element_i<N, double> &elem =
			sym.get_element(ielem);
		index<N> idx2(idx);
		transf<N, double> tr2(tr);
		if(elem.is_allowed(idx2)) {
			elem.apply(idx2, tr2);
			allowed = make_transf_map(sym, bidims,
				idx2, tr2, alltransf);
		} else {
			allowed = false;
		}
	}
	return allowed;
}


template<size_t N>
void btod_random<N>::make_random_blk(block_tensor_ctrl<N, double> &ctrl,
	const dimensions<N> &bidims, const index<N> &idx) {

	typedef libvmm::std_allocator<double> allocator_t;

	const symmetry<N, double> &sym = ctrl.req_symmetry();
	size_t absidx = bidims.abs_index(idx);
	tod_random<N> randop;

	transf<N, double> tr0;
	transf_map_t transf_map;
	bool allowed = make_transf_map(sym, bidims, idx, tr0, transf_map);
	typename transf_map_t::iterator ilst = transf_map.find(absidx);
	if(!allowed || ilst == transf_map.end()) {
		ctrl.req_zero_block(idx);
		return;
	}

	tensor_i<N, double> &blk = ctrl.req_block(idx);

	typename transf_list_t::iterator itr = ilst->second.begin();
	if(itr == ilst->second.end()) {
		timings_base::start_timer("randop");
		randop.perform(blk);
		timings_base::stop_timer("randop");
	} else {
		tensor<N, double, allocator_t> rnd(blk.get_dims()),
			symrnd(blk.get_dims());
		timings_base::start_timer("randop");
		randop.perform(rnd);
		timings_base::stop_timer("randop");
		double totcoeff = itr->get_coeff();
		tod_add<N> symop(rnd, itr->get_perm(), totcoeff);

		for(itr++; itr != ilst->second.end(); itr++) {
			symop.add_op(rnd, itr->get_perm(), itr->get_coeff());
			totcoeff += itr->get_coeff();
		}

		timings_base::start_timer("symop&copy");
		symop.perform(symrnd);
		totcoeff = (totcoeff == 0.0) ? 1.0 : 1.0/totcoeff;
		tod_copy<N>(symrnd, totcoeff).perform(blk);
		timings_base::stop_timer("symop&copy");
	}

	ctrl.ret_block(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_H
