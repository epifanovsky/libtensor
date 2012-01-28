#ifndef LIBTENSOR_BTOD_EXTRACT_H
#define LIBTENSOR_BTOD_EXTRACT_H

#include "../defs.h"
#include "../not_implemented.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_proj_down.h"
#include "../symmetry/so_permute.h"
#include "../tod/tod_extract.h"
#include <libtensor/dense_tensor/tod_set.h>
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "transf_double.h"

#include "../core/block_index_space.h"

namespace libtensor {


/**	\brief Extracts a tensor with smaller dimension from the %tensor
	\tparam N Tensor order.
	\tparam M Number of fixed dimensions.
	\tparam N - M result tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_extract :
	public additive_btod<N - M>,
	public timings< btod_extract<N, M> > {

public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the argument
	static const size_t k_orderb = N - M; //!< Order of the result

private:
	block_tensor_i<N, double> &m_bta; //!< Input block %tensor
	mask<N> m_msk;//!< Mask for extraction
	permutation<k_orderb> m_perm; //!< Permutation of the result
	double m_c; //!< Scaling coefficient
	block_index_space<k_orderb> m_bis; //!< Block %index space of the result
	index<N> m_idxbl;//!< Index for extraction of the block
	index<N> m_idxibl;//!< Index for extraction inside the block
	symmetry<k_orderb, double> m_sym; //!< Symmetry of the result
	assignment_schedule<k_orderb, double> m_sch; //!< Assignment schedule

public:
	btod_extract(block_tensor_i<N, double> &bta, const mask<N> &m,
		const index<N> &idxbl, const index<N> &idxibl, double c = 1.0);

	btod_extract(block_tensor_i<N, double> &bta, const mask<N> &m,
		const permutation<N - M> &perm, const index<N> &idxbl,
		const index<N> &idxibl, double c = 1.0);

	//!	\name Implementation of
	//		libtensor::direct_tensor_operation<N - M + 1, double>
	//@{

	virtual const block_index_space<k_orderb> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<k_orderb, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<k_orderb, double> &get_schedule() const {
		return m_sch;
	}

	//@}

	using additive_btod<k_orderb>::perform;

	virtual void sync_on();
	virtual void sync_off();

protected:
	virtual void compute_block(bool zero, dense_tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i, const transf<k_orderb, double> &tr,
		double c, cpu_pool &cpus);

private:
	/**	\brief Forms the block %index space of the output or throws an
			exception if the input is incorrect
	 **/
	static block_index_space<N - M> mk_bis(const block_index_space<N> &bis,
		const mask<N> &msk, const permutation<N - M> &perm);

	/**	\brief Sets up the assignment schedule for the operation.
	 **/
	void make_schedule();

	void do_compute_block(dense_tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i, const transf<k_orderb, double> &tr,
		double c, bool zero, cpu_pool &cpus);

private:
	btod_extract(const btod_extract<N, M>&);
	const btod_extract<N, M> &operator=(const btod_extract<N, M>&);

};


template<size_t N, size_t M>
const char *btod_extract<N, M>::k_clazz = "btod_extract<N, M>";


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
	const mask<N> &m, const index<N> &idxbl, const index<N> &idxibl,
	double c) :

	m_bta(bta), m_msk(m), m_idxbl(idxbl), m_idxibl(idxibl), m_c(c),
	m_bis(mk_bis(bta.get_bis(), m_msk, permutation<N - M>())), m_sym(m_bis),
	m_sch(m_bis.get_block_index_dims()) {

	block_tensor_ctrl<N, double> ctrla(bta);
	so_proj_down<N, M, double>(ctrla.req_const_symmetry(), m_msk).
		perform(m_sym);

	make_schedule();
}


template<size_t N, size_t M>
btod_extract<N, M>::btod_extract(block_tensor_i<N, double> &bta,
	const mask<N> &m, const permutation<N - M> &perm,
	const index<N> &idxbl, const index<N> &idxibl, double c) :

	m_bta(bta), m_msk(m), m_perm(perm), m_idxbl(idxbl), m_idxibl(idxibl),
	m_c(c), m_bis(mk_bis(bta.get_bis(), m_msk, perm)), m_sym(m_bis),
	m_sch(m_bis.get_block_index_dims()) {

	permutation<N - M> pinv(perm, true);
	block_index_space<N - M> bisinv(m_bis);
	bisinv.permute(pinv);

	block_tensor_ctrl<N, double> ctrla(bta);
	symmetry<k_orderb, double> sym(bisinv);
	so_proj_down<N, M, double>(ctrla.req_const_symmetry(), m_msk).
		perform(sym);
	so_permute<k_orderb, double>(sym, perm).perform(m_sym);

	make_schedule();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_on() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	ctrla.req_sync_on();
}


template<size_t N, size_t M>
void btod_extract<N, M>::sync_off() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	ctrla.req_sync_off();
}

/*
template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(dense_tensor_i<k_orderb, double> &blk,
	const index<k_orderb> &idx) {

	transf<k_orderb, double> tr0;
	do_compute_block(blk, idx, tr0, 1.0, true);
}*/


template<size_t N, size_t M>
void btod_extract<N, M>::compute_block(bool zero, dense_tensor_i<k_orderb, double> &blk,
	const index<k_orderb> &idx, const transf<k_orderb, double> &tr,
	double c, cpu_pool &cpus) {

	do_compute_block(blk, idx, tr, c, zero, cpus);
}


template<size_t N, size_t M>
void btod_extract<N, M>::do_compute_block(dense_tensor_i<k_orderb, double> &blk,
	const index<k_orderb> &idx, const transf<k_orderb, double> &tr,
	double c, bool zero, cpu_pool &cpus) {

	btod_extract<N, M>::start_timer();

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);

	permutation<k_orderb> pinv(m_perm, true);

	index<k_ordera> idxa;
	index<k_orderb> idxb(idx);

	idxb.permute(pinv);

	for(size_t i = 0, j = 0; i < k_ordera; i++) {
		if(m_msk[i]) {
			idxa[i] = idxb[j++];
		} else {
			idxa[i] = m_idxbl[i];
		}
	}

	orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

	abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
			m_bta.get_bis().get_block_index_dims());
	transf<k_ordera, double> tra(oa.get_transf(idxa)); tra.invert();

	mask<k_ordera> msk1(m_msk), msk2(m_msk);
	msk2.permute(tra.get_perm());

	sequence<k_ordera, size_t> seqa1(0), seqa2(0);
	sequence<k_orderb, size_t> seqb1(0), seqb2(0);
	for(register size_t i = 0; i < k_ordera; i++) seqa2[i] = seqa1[i] = i;
	tra.get_perm().apply(seqa2);
	for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
		if(msk1[i]) seqb1[j1++] = seqa1[i];
		if(msk2[i]) seqb2[j2++] = seqa2[i];
	}

	permutation_builder<k_orderb> pb(seqb2, seqb1);
	permutation<k_orderb> permb(pb.get_perm());
	permb.permute(m_perm);
	permb.permute(tr.get_perm());

	index<k_ordera> idxibl2(m_idxibl);
	idxibl2.permute(tra.get_perm());

	bool zeroa = !oa.is_allowed();
	if(!zeroa) zeroa = ctrla.req_is_zero_block(cidxa.get_index());

	if(!zeroa) {

		dense_tensor_i<k_ordera, double> &blka = ctrla.req_block(
			cidxa.get_index());
		if(zero) {
			tod_extract<N, M>(blka, msk2, permb, idxibl2,
				tra.get_coeff() * m_c * c).perform(blk);
		} else {
			tod_extract<N, M>(blka, msk2, permb, idxibl2,
				tra.get_coeff() * m_c).perform(blk, c);
		}
		ctrla.ret_block(cidxa.get_index());
	} else {

		if(zero) tod_set<N - M>().perform(cpus, blk);
	}

	btod_extract<N, M>::stop_timer();
}


template<size_t N, size_t M>
block_index_space<N - M> btod_extract<N, M>::mk_bis(
	const block_index_space<N> &bis, const mask<N> &msk,
	const permutation<N - M> &perm) {

	static const char *method = "mk_bis(const block_index_space<N>&, "
		"const mask<N>&, const permutation<N - M>&)";

	dimensions<N> idims(bis.get_dims());

	//	Compute output dimensions
	//

	index<k_orderb> i1, i2;

	size_t m = 0, j = 0;
	size_t map[k_orderb];//map between B and A

	for(size_t i = 0; i < N; i++) {
		if(msk[i]) {
			i2[j] = idims[i] - 1;
			map[j] = i;
			j++;
		} else {
			m++;
		}
	}


	if(m != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"m");
	}

	block_index_space<k_orderb> obis(dimensions<k_orderb>(
		index_range<k_orderb>(i1, i2)));

	mask<k_orderb> msk_done;
	bool done = false;
	while(!done) {

		size_t i = 0;
		while(i < k_orderb && msk_done[i]) i++;
		if(i == k_orderb) {
			done = true;
			continue;
		}
		size_t typ = bis.get_type(map[i]);
		const split_points &splits = bis.get_splits(typ);
		mask<k_orderb> msk_typ;
		for(size_t k = 0; k < k_orderb; k++) {
			if(bis.get_type(map[k]) == typ) msk_typ[k] = true;
		}
		size_t npts = splits.get_num_points();
		for(register size_t k = 0; k < npts; k++) {
			obis.split(msk_typ, splits[k]);
		}
		msk_done |= msk_typ;
	}

	obis.permute(perm);
	return obis;
}


template<size_t N, size_t M>
void btod_extract<N, M>::make_schedule() {

	btod_extract<N, M>::start_timer("make_schedule");

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<k_orderb> pinv(m_perm, true);

	orbit_list<k_orderb, double> olb(m_sym);
	for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
			iob != olb.end(); iob++) {

		index<k_ordera> idxa;
		index<k_orderb> idxb(olb.get_index(iob));

		idxb.permute(pinv);

		for(size_t i = 0, j = 0; i < k_ordera; i++) {
			if(m_msk[i]) idxa[i] = idxb[j++];
			else idxa[i] = m_idxbl[i];
		}

		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);

		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
				m_bta.get_bis().get_block_index_dims());

		if(!oa.is_allowed()) continue;
		if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

		m_sch.insert(olb.get_abs_index(iob));
	}

	btod_extract<N, M>::stop_timer("make_schedule");

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_H
