#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include "../symmetry/so_copy.h"
#include "../symmetry/so_permute.h"
#include "../symmetry/so_proj_down.h"
#include "../symmetry/so_proj_up.h"
#include "../symmetry/so_union.h"
#include "../tod/tod_contract2.h"
#include "../tod/tod_sum.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) :

	m_contr(contr), m_bta(bta), m_btb(btb),
	m_sym_bld(m_contr, m_bta, m_btb),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_sym_bld.get_bis().get_block_index_dims()),
	m_sch(m_bidimsc) {

	make_schedule();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {

	clear_schedule(m_contr_sch);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::sync_on() {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	ctrla.req_sync_on();
	ctrlb.req_sync_on();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::sync_off() {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	ctrla.req_sync_off();
	ctrlb.req_sync_off();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(tensor_i<N + M, double> &blk,
	const index<N + M> &i) {

	static const char *method =
		"compute_block(tensor_i<N + M, double>&, const index<N + M>&)";

	btod_contract2<N, M, K>::start_timer();

	try {

		block_tensor_ctrl<k_ordera, double> ca(m_bta);
		block_tensor_ctrl<k_orderb, double> cb(m_btb);

		abs_index<k_orderc> aic(i, m_bidimsc);
		typename schedule_t::iterator isch =
			m_contr_sch.find(aic.get_abs_index());
		if(isch == m_contr_sch.end()) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "i");
		}

		transf<k_orderc, double> trc0;
		contract_block(*isch->second, aic.get_index(), ca, cb,
			blk, trc0, true, 1.0);

	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(tensor_i<N + M, double> &blk,
	const index<N + M> &i, const transf<N + M, double> &tr, double c) {

	static const char *method = "compute_block(tensor_i<N + M, double>&, "
		"const index<N + M>&, const transf<N + M, double>&, double)";

	btod_contract2<N, M, K>::start_timer();

	try {

		block_tensor_ctrl<k_ordera, double> ca(m_bta);
		block_tensor_ctrl<k_orderb, double> cb(m_btb);

		abs_index<k_orderc> aic(i, m_bidimsc);
		typename schedule_t::iterator isch =
			m_contr_sch.find(aic.get_abs_index());
		if(isch == m_contr_sch.end()) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "i");
		}

		contract_block(*isch->second, aic.get_index(), ca, cb,
			blk, tr, false, c);
	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

	btod_contract2<N, M, K>::start_timer("make_schedule");

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	dimensions<k_ordera> bidimsa(m_bta.get_bis().get_block_index_dims());
	dimensions<k_orderb> bidimsb(m_btb.get_bis().get_block_index_dims());

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	orbit_list<k_orderc, double> olc(get_symmetry());

	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		orbit<k_ordera, double> oa(ca.req_const_symmetry(),
			ola.get_index(ioa));
		if(!oa.is_allowed()) continue;
		if(ca.req_is_zero_block(ola.get_index(ioa))) continue;

		abs_index<k_ordera> acia(ola.get_index(ioa), bidimsa);
		for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
			ia != oa.end(); ia++) {

			abs_index<k_ordera> aia(oa.get_abs_index(ia), bidimsa);
			make_schedule_a(cb, bidimsb, olc, m_bidimsc, aia, acia,
				oa.get_transf(ia));
		}
	}

	btod_contract2<N, M, K>::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_a(
	block_tensor_ctrl<k_orderb, double> &cb,
	const dimensions<k_orderb> &bidimsb,
	const orbit_list<k_orderc, double> &olc,
	const dimensions<k_orderc> &bidimsc, const abs_index<k_ordera> &aia,
	const abs_index<k_ordera> &acia, const transf<k_ordera, double> &tra) {

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	const index<k_ordera> &ia = aia.get_index();

	sequence<M, size_t> useqb(0), useqc(0); // Maps uncontracted indexes
	index<M> iu1, iu2;
	for(size_t i = 0, j = 0; i < k_orderc; i++) {
		if(conn[i] >= k_orderc + k_ordera) {
			useqb[j] = conn[i] - k_orderc - k_ordera;
			useqc[j] = i;
			j++;
		}
	}
	for(size_t i = 0; i < M; i++) iu2[i] = bidimsb[useqb[i]] - 1;
	// Uncontracted indexes from B
	dimensions<M> bidimsu(index_range<M>(iu1, iu2));
	abs_index<M> aiu(bidimsu);
	do {
		const index<M> &iu = aiu.get_index();

		// Construct the index in C
		index<k_orderc> ic;
		for(size_t i = 0, j = 0; i < k_orderc; i++) {
			if(conn[i] < k_orderc + k_ordera) {
				ic[i] = ia[conn[i] - k_orderc];
			} else {
				ic[i] = iu[j++];
			}
		}

		// Skip non-canonical indexes in C
		abs_index<k_orderc> aic(ic, bidimsc);
		if(!olc.contains(aic.get_abs_index())) continue;

		// Construct the index in B
		index<k_orderb> ib;
		for(size_t i = 0; i < k_orderb; i++) {
			register size_t k = conn[k_orderc + k_ordera + i];
			if(k < k_orderc) ib[i] = ic[k];
			else ib[i] = ia[k - k_orderc];
		}
		make_schedule_b(cb, bidimsb, acia, tra, ib, aic);
	} while(aiu.inc());
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_b(
	block_tensor_ctrl<k_orderb, double> &cb,
	const dimensions<k_orderb> &bidimsb, const abs_index<k_ordera> &acia,
	const transf<k_ordera, double> &tra, const index<k_orderb> &ib,
	const abs_index<k_orderc> &acic) {

	orbit<k_orderb, double> ob(cb.req_const_symmetry(), ib);
	if(!ob.is_allowed()) return;

	abs_index<k_orderb> acib(ob.get_abs_canonical_index(), bidimsb);
	if(cb.req_is_zero_block(acib.get_index())) return;

	const transf<k_orderb, double> &trb = ob.get_transf(ib);
	block_contr_t bc(acia.get_abs_index(), acib.get_abs_index(),
		tra.get_coeff() * trb.get_coeff(), tra.get_perm(),
		trb.get_perm());
	schedule_block_contraction(acic, bc);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::schedule_block_contraction(
	const abs_index<k_orderc> &acic, const block_contr_t &bc) {

	// Check whether this is the first contraction for this block in C
	typename schedule_t::iterator isch =
		m_contr_sch.find(acic.get_abs_index());
	if(isch == m_contr_sch.end()) {
		m_sch.insert(acic.get_abs_index());
		block_contr_list_t *lst = new block_contr_list_t;
		lst->push_back(bc);
		m_contr_sch.insert(std::pair<size_t, block_contr_list_t*>(
			acic.get_abs_index(), lst));
		return;
	}

	block_contr_list_t &lst = *isch->second;

	// Find similar contractions already on the list
	typename block_contr_list_t::iterator ilst = lst.begin();
	while(ilst != lst.end() && ilst->m_absidxa < bc.m_absidxa) ilst++;
	while(ilst != lst.end() && ilst->m_absidxa == bc.m_absidxa &&
		ilst->m_absidxb < bc.m_absidxb) ilst++;

	// If similar contractions are found, try to combine with them
	bool done = false;
	while(!done && ilst != lst.end() && ilst->m_absidxa == bc.m_absidxa &&
		ilst->m_absidxb == bc.m_absidxb) {

		if(ilst->m_perma.equals(bc.m_perma) &&
			ilst->m_permb.equals(bc.m_permb)) {
			ilst->m_c += bc.m_c;
			if(ilst->m_c == 0.0) {
				lst.erase(ilst);
				ilst = lst.end();
			}
			done = true;
		} else {
			ilst++;
		}
	}

	// If similar contractions are not found, simply add bc
	if(!done) lst.insert(ilst, bc);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::clear_schedule(schedule_t &sch) {

	typename schedule_t::iterator isch = sch.begin();
	for(; isch != sch.end(); isch++) {
		delete isch->second;
		isch->second = 0;
	}
	sch.clear();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
	block_contr_list_t &lst, const index<k_orderc> &idxc,
	block_tensor_ctrl<k_ordera, double> &ca,
	block_tensor_ctrl<k_orderb, double> &cb,
	tensor_i<k_orderc, double> &tc, const transf<k_orderc, double> &trc,
	bool zero, double c) {

	if(zero) tod_set<k_orderc>().perform(tc);

	std::list< index<k_ordera> > blksa;
	std::list< index<k_orderb> > blksb;
	std::list< tod_contract2<N, M, K>* > op_ptrs;
	tod_sum<k_orderc> *op_sum = 0;

	for(typename block_contr_list_t::iterator ilst = lst.begin();
		ilst != lst.end(); ilst++) {

		abs_index<k_ordera> aia(ilst->m_absidxa, m_bidimsa);
		abs_index<k_orderb> aib(ilst->m_absidxb, m_bidimsb);
		const index<k_ordera> &ia = aia.get_index();
		const index<k_orderb> &ib = aib.get_index();

		bool zeroa = ca.req_is_zero_block(ia);
		bool zerob = cb.req_is_zero_block(ib);
		if(zeroa || zerob) continue;

		tensor_i<k_ordera, double> &blka = ca.req_block(ia);
		tensor_i<k_orderb, double> &blkb = cb.req_block(ib);
		blksa.push_back(ia);
		blksb.push_back(ib);

		contraction2<N, M, K> contr(m_contr);
		contr.permute_a(ilst->m_perma);
		contr.permute_b(ilst->m_permb);
		contr.permute_c(trc.get_perm());

		tod_contract2<N, M, K> *controp =
			new tod_contract2<N, M, K>(contr, blka, blkb);
		double kc = ilst->m_c * trc.get_coeff();
		op_ptrs.push_back(controp);
		if(op_sum == 0) op_sum = new tod_sum<k_orderc>(*controp, kc);
		else op_sum->add_op(*controp, kc);
	}

	if(op_sum != 0) {
		op_sum->prefetch();
		op_sum->perform(tc, c);
		delete op_sum; op_sum = 0;
		for(typename std::list< tod_contract2<N, M, K>* >::const_iterator iptr =
			op_ptrs.begin(); iptr != op_ptrs.end(); iptr++) {

			delete *iptr;
		}
		op_ptrs.clear();
	}

	for(typename std::list< index<k_ordera> >::const_iterator i =
		blksa.begin(); i != blksa.end(); i++) ca.ret_block(*i);
	for(typename std::list< index<k_orderb> >::const_iterator i =
		blksb.begin(); i != blksb.end(); i++) cb.ret_block(*i);
}


template<size_t N, size_t M, size_t K>
btod_contract2_symmetry_builder<N, M, K>::btod_contract2_symmetry_builder(
	const contraction2<N, M, K> &contr,
	block_tensor_i<N + K, double> &bta,
	block_tensor_i<M + K, double> &btb) :

	btod_contract2_symmetry_builder_base<N, M, K>(
		contr, make_xbis(bta.get_bis(), btb.get_bis())) {

	make_symmetry(contr, bta, btb);
}


template<size_t N, size_t K>
btod_contract2_symmetry_builder<N, N, K>::btod_contract2_symmetry_builder(
	const contraction2<N, N, K> &contr,
	block_tensor_i<N + K, double> &bta,
	block_tensor_i<N + K, double> &btb) :

	btod_contract2_symmetry_builder_base<N, N, K>(contr,
		&bta == &btb ? make_xbis(bta.get_bis()) :
			btod_contract2_symmetry_builder_base<N, N, K>::
				make_xbis(bta.get_bis(), btb.get_bis())) {

	if(&bta == &btb) make_symmetry(contr, bta);
	else btod_contract2_symmetry_builder_base<N, N, K>::make_symmetry(
		contr, bta, btb);
}


template<size_t N, size_t K>
block_index_space<2 * (N + K)>
btod_contract2_symmetry_builder<N, N, K>::make_xbis(
	const block_index_space<N + K> &bisa) {

	const dimensions<N + K> &dimsa = bisa.get_dims();

	index<2 * (N + K)> xi1, xi2;
	for(size_t i = 0; i < N + K; i++) {
		xi2[N + K + i] = xi2[i] = dimsa[i] - 1;
	}
	dimensions<2 * (N + K)> xdimsab(index_range<2 * (N + K)>(xi1, xi2));
	block_index_space<2 * (N + K)> xbisab(xdimsab);

	mask<N + K> ma;

	size_t ia = 0;
	while(true) {
		while(ia < N + K && ma[ia]) ia++;
		if(ia == N + K) break;
		size_t typ = bisa.get_type(ia);

		mask<N + K> ma_split;
		mask<2 * (N + K)> mab_split;
		for(size_t i = ia; i < N + K; i++) {
			mab_split[N + K + i] = mab_split[i] = ma_split[i] =
				bisa.get_type(i) == typ;
		}

		const split_points &pts = bisa.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			xbisab.split(mab_split, pts[ipt]);

		ma |= ma_split;
	}

	return xbisab;
}


template<size_t N, size_t K>
void btod_contract2_symmetry_builder<N, N, K>::make_symmetry(
	const contraction2<N, N, K> &contr,
	block_tensor_i<N + K, double> &bta) {

	block_tensor_ctrl<N + K, double> ca(bta);
	const sequence<2 * (2 * N + K), size_t> &conn = contr.get_conn();
	const block_index_space<2 * (N + K)> &xbis =
		btod_contract2_symmetry_builder_base<N, N, K>::get_xbis();
	const block_index_space<2 * N> &bis =
		btod_contract2_symmetry_builder_base<N, N, K>::get_bis();

	symmetry<2 * (N + K), double> xsyma(xbis), xsymb(xbis), xsymab(xbis);
	mask<2 * (N + K)> xma, xmb;
	permutation<N + K> perma0;
	for(size_t i = 0; i < N + K; i++) {
		xma[i] = true;
		xmb[N + K + i] = true;
	}

	so_proj_up<N + K, N + K, double>(ca.req_const_symmetry(), perma0, xma).
		perform(xsyma);
	so_proj_up<N + K, N + K, double>(ca.req_const_symmetry(), perma0, xmb).
		perform(xsymb);
	so_union<2 * (N + K), double>(xsyma, xsymb).perform(xsymab);

	//	When a tensor is contracted with itself, there is additional
	//	perm symmetry

	permutation<2 * (N + K)> permab;
	for(size_t i = 0; i < N + K; i++) {
		permab.permute(i, N + K + i);
	}
	if(!permab.is_identity()) {
		xsymab.insert(se_perm<2 * (N + K), double>(permab, true));
	}
	//~ for(size_t i = 0; i < N + K; i++) {
		//~ if(conn[2 * N + i] > 3 * N + K) {
			//~ permutation<2 * (N + K)> permab2;
			//~ permab2.permute(i, conn[2 * N + i] - 2 * N);
			//~ xsymab.insert(se_perm<2 * (N + K), double>(
				//~ permab2, true));
		//~ }
	//~ }

	mask<2 * (N + K)> rmab;
	sequence<2 * N, size_t> seq1(0), seq2(0);
	for(size_t i = 0, j = 0; i < 2 * (N + K); i++) {
		if(conn[2 * N + i] < 2 * N) {
			rmab[i] = true;
			seq1[j] = conn[j];
			seq2[j] = 2 * N + i;
			j++;
		}
	}

	permutation_builder<2 * N> rperm_bld(seq1, seq2);
	symmetry<2 * N, double> sym(bis);
	so_proj_down<2 * (N + K), 2 * K, double>(xsymab, rmab).
		perform(sym);
	so_permute<2 * N, double>(sym, rperm_bld.get_perm()).perform(
		btod_contract2_symmetry_builder_base<N, N, K>::get_symmetry());
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M + 2 * K>
btod_contract2_symmetry_builder_base<N, M, K>::make_xbis(
	const block_index_space<N + K> &bisa,
	const block_index_space<M + K> &bisb) {

	const dimensions<N + K> &dimsa = bisa.get_dims();
	const dimensions<M + K> &dimsb = bisb.get_dims();

	index<N + M + 2 * K> xi1, xi2;
	for(size_t i = 0; i < N + K; i++) xi2[i] = dimsa[i] - 1;
	for(size_t i = 0; i < M + K; i++) xi2[N + K + i] = dimsb[i] - 1;
	dimensions<N + M + 2 * K> xdimsab(index_range<N + M + 2 * K>(xi1, xi2));
	block_index_space<N + M + 2 * K> xbisab(xdimsab);

	mask<N + K> ma;
	mask<M + K> mb;

	size_t ia = 0, ib = 0;
	while(true) {
		while(ia < N + K && ma[ia]) ia++;
		if(ia == N + K) break;
		size_t typ = bisa.get_type(ia);

		mask<N + K> ma_split;
		mask<N + M + 2 * K> mab_split;
		for(size_t i = ia; i < N + K; i++) {
			mab_split[i] = ma_split[i] = bisa.get_type(i) == typ;
		}

		const split_points &pts = bisa.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			xbisab.split(mab_split, pts[ipt]);

		ma |= ma_split;
	}
	while(true) {
		while(ib < M + K && mb[ib]) ib++;
		if(ib == M + K) break;
		size_t typ = bisb.get_type(ib);

		mask<M + K> mb_split;
		mask<N + M + 2 * K> mab_split;
		for(size_t i = ib; i < M + K; i++) {
			mab_split[N + K + i] = mb_split[i] =
				bisb.get_type(i) == typ;
		}

		const split_points &pts = bisb.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			xbisab.split(mab_split, pts[ipt]);

		mb |= mb_split;
	}

	return xbisab;
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M>
btod_contract2_symmetry_builder_base<N, M, K>::make_bis(
	const contraction2<N, M, K> &contr,
	const block_index_space<N + M + 2 * K> &xbis) {

	const dimensions<N + M + 2 * K> &xdims = xbis.get_dims();
	const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

	index<N + M> i1, i2;
	for(size_t i = 0; i < N + M; i++) i2[i] = xdims[conn[i] - N - M] - 1;
	dimensions<N + M> dims(index_range<N + M>(i1, i2));
	block_index_space<N + M> bis(dims);

	mask<N + M> mc;

	size_t ic = 0;
	while(true) {
		while(ic < N + M && mc[ic]) ic++;
		if(ic == N + M) break;
		size_t typ = xbis.get_type(conn[ic] - N - M);

		mask<N + M> mc_split;
		for(size_t i = ic; i < N + M; i++) {
			mc_split[i] = xbis.get_type(conn[i] - N - M) == typ;
		}

		const split_points &pts = xbis.get_splits(typ);
		size_t npts = pts.get_num_points();
		for(size_t ipt = 0; ipt < npts; ipt++)
			bis.split(mc_split, pts[ipt]);

		mc |= mc_split;
	}

	bis.match_splits();
	return bis;
}


template<size_t N, size_t M, size_t K>
void btod_contract2_symmetry_builder_base<N, M, K>::make_symmetry(
	const contraction2<N, M, K> &contr,
	block_tensor_i<N + K, double> &bta,
	block_tensor_i<M + K, double> &btb) {

	block_tensor_ctrl<N + K, double> ca(bta);
	block_tensor_ctrl<M + K, double> cb(btb);
	const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

	symmetry<N + M + 2 * K, double> xsyma(m_xbis), xsymb(m_xbis),
		xsymab(m_xbis);
	mask<N + M + 2 * K> xma, xmb;
	permutation<N + K> perma0;
	permutation<M + K> permb0;
	for(size_t i = 0; i < N + K; i++) xma[i] = true;
	for(size_t i = 0; i < M + K; i++) xmb[N + K + i] = true;

	so_proj_up<N + K, M + K, double>(ca.req_const_symmetry(), perma0, xma).
		perform(xsyma);
	so_proj_up<M + K, N + K, double>(cb.req_const_symmetry(), permb0, xmb).
		perform(xsymb);
	so_union<N + M + 2 * K, double>(xsyma, xsymb).perform(xsymab);

	mask<N + M + 2 * K> rmab;
	sequence<N + M, size_t> seq1(0), seq2(0);
	for(size_t i = 0, j = 0; i < N + M + 2 * K; i++) {
		if(conn[N + M + i] < N + M) {
			rmab[i] = true;
			seq1[j] = conn[j];
			seq2[j] = N + M + i;
			j++;
		}
	}

	permutation_builder<N + M> rperm_bld(seq1, seq2);
	symmetry<N + M, double> sym(m_bis);
	so_proj_down<N + M + 2 * K, 2 * K, double>(xsymab, rmab).
		perform(sym);
	so_permute<N + M, double>(sym, rperm_bld.get_perm()).perform(m_sym);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
