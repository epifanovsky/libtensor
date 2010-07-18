#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include "../symmetry/so_concat.h"
#include "../symmetry/so_proj_down.h"
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

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	orbit_list<k_orderb, double> olb(cb.req_const_symmetry());
	orbit_list<k_orderc, double> olc(get_symmetry());

	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		if(ca.req_is_zero_block(ola.get_index(ioa))) continue;

		orbit<k_ordera, double> oa(ca.req_const_symmetry(),
			ola.get_index(ioa));

		for(typename orbit_list<k_orderb, double>::iterator iob =
			olb.begin(); iob != olb.end(); iob++) {

			if(cb.req_is_zero_block(olb.get_index(iob))) continue;

			orbit<k_orderb, double> ob(cb.req_const_symmetry(),
				olb.get_index(iob));

				make_schedule(oa, ob, olc);
		}
	}

	btod_contract2<N, M, K>::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule(const orbit<k_ordera, double> &oa,
	const orbit<k_orderb, double> &ob,
	const orbit_list<k_orderc, double> &olc) {

	sequence<K, size_t> ka(0), kb(0);
	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	for(size_t i = 0, j = 0; i < k_ordera; i++) {
		if(conn[k_orderc + i] >= k_orderc + k_ordera) {
			ka[j] = i;
			kb[j] = conn[k_orderc + i] - k_orderc - k_ordera;
			j++;
		}
	}

	typedef std::multimap<size_t, block_contr_t> local_schedule_t;
	local_schedule_t local_sch;

	for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
		ia != oa.end(); ia++) {

	abs_index<k_ordera> aidxa(oa.get_abs_index(ia), m_bidimsa);
	const index<k_ordera> &idxa = aidxa.get_index();
	const transf<k_ordera, double> &tra = oa.get_transf(
		aidxa.get_abs_index());

	for(typename orbit<k_orderb, double>::iterator ib = ob.begin();
		ib != ob.end(); ib++) {

		abs_index<k_orderb> aidxb(ob.get_abs_index(ib), m_bidimsb);
		const index<k_orderb> &idxb = aidxb.get_index();
		const transf<k_orderb, double> &trb = ob.get_transf(
			aidxb.get_abs_index());

		bool need_contr = true;
		for(size_t i = 0; i < K; i++) {
			if(idxa[ka[i]] != idxb[kb[i]]) {
				need_contr = false;
				break;
			}
		}
		if(!need_contr) continue;

		index<k_orderc> idxc;
		for(size_t i = 0; i < k_orderc; i++) {
			register size_t j = conn[i] - k_orderc;
			idxc[i] = j < k_ordera ? idxa[j] : idxb[j - k_ordera];
		}
		abs_index<k_orderc> aidxc(idxc, m_bidimsc);

		if(!olc.contains(aidxc.get_abs_index())) continue;

		std::pair<typename local_schedule_t::iterator,
			typename local_schedule_t::iterator> itpair =
				local_sch.equal_range(aidxc.get_abs_index());
		bool done = false;
		for(typename local_schedule_t::iterator isch = itpair.first;
			isch != itpair.second; isch++) {

			block_contr_t &bc = isch->second;
			if(bc.is_same_perm(tra, trb)) {
				bc.m_c += tra.get_coeff() * trb.get_coeff();
				done = true;
				break;
			}
		}
		if(!done) {
			block_contr_t bc(oa.get_abs_canonical_index(),
				ob.get_abs_canonical_index(),
				tra.get_coeff() * trb.get_coeff(),
				tra.get_perm(), trb.get_perm());
			local_sch.insert(std::pair<size_t, block_contr_t>(
				aidxc.get_abs_index(), bc));
		}

	} // for ib

	} // for ia

	typename local_schedule_t::iterator ilocsch = local_sch.begin();
	for(; ilocsch != local_sch.end(); ilocsch++) {

		block_contr_t &bc = ilocsch->second;
		if(bc.m_c == 0.0) continue;
		typename schedule_t::iterator isch =
			m_contr_sch.find(ilocsch->first);
		if(isch == m_contr_sch.end()) {
			block_contr_list_t *lst = new block_contr_list_t;
			lst->push_back(bc);
			m_contr_sch.insert(std::pair<size_t, block_contr_list_t*>(
				ilocsch->first, lst));
		} else {
			isch->second->push_back(bc);
		}
		if(!m_sch.contains(ilocsch->first)) {
			m_sch.insert(ilocsch->first);
		}
	}

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
	block_index_space<2 * (N + K)> xbis(
		btod_contract2_symmetry_builder_base<N, N, K>::get_xbis());
	const block_index_space<2 * N> &bis =
		btod_contract2_symmetry_builder_base<N, N, K>::get_bis();

	sequence<2 * (N + K), size_t> seq1(0), seq2(0);
	mask<2 * (N + K)> rmab;
	for (size_t i = 0, j = 0; i < 2 * (N + K); i++) {
		seq1[i] = i;
		if (conn[i + 2 * N] < 2 * N) {
			seq2[conn[i + 2 * N]] = i;
			rmab[conn[i + 2 * N]] = true;
		}
		else {
			seq2[2 * N + j] = i;
			j++;
		}
	}

	permutation_builder<2 * (N + K)> pb(seq2, seq1);
	xbis.permute(pb.get_perm());
	symmetry<2 * (N + K), double> xsymab(xbis);


	so_concat<N + K, N + K, double>(ca.req_const_symmetry(),
			ca.req_const_symmetry(), pb.get_perm()).perform(xsymab);

	//	When a tensor is contracted with itself, there is additional
	//	perm symmetry

	permutation<2 * (N + K)> permab(pb.get_perm(), true);
	for(size_t i = 0; i < N + K; i++) {
		permab.permute(i, N + K + i);
	}
	permab.permute(pb.get_perm());
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


	so_proj_down<2 * (N + K), 2 * K, double>(xsymab, rmab).perform(
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

	symmetry<N + M + 2 * K, double> xsymab(m_xbis);

	sequence<N + M + 2 * K, size_t> seq1(0), seq2(0);
	mask<N + M + 2 * K> rmab;
	for (size_t i = 0, j = 0; i < N + M + 2 * K; i++) {
		seq1[i] = i;
		if (conn[i + N + M] < N + M) {
			seq2[conn[i + N + M]] = i;
			rmab[conn[i + N + M]] = true;
		}
		else {
			seq2[N + M + j] = i;
			j++;
		}
	}
	permutation_builder<N + M + 2 * K> pb(seq2, seq1);

	so_concat<N + K, M + K, double>(ca.req_const_symmetry(),
			cb.req_const_symmetry(), pb.get_perm()).perform(xsymab);

	so_proj_down<N + M + 2 * K, 2 * K, double>(xsymab, rmab).perform(
			btod_contract2_symmetry_builder_base<N, M, K>::get_symmetry());


}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
