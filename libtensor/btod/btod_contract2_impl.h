#ifndef LIBTENSOR_BTOD_CONTRACT2_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT2_IMPL_H

#include <libtensor/mp/auto_cpu_lock.h>
#include "../core/block_index_subspace_builder.h"
#include "../core/mask.h"
#include "../symmetry/so_concat.h"
#include "../symmetry/so_stabilize.h"
#include <libtensor/dense_tensor/tod_contract2.h>
#include "../tod/tod_sum.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_contract2_clazz<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz =
	btod_contract2_clazz<N, M, K>::k_clazz;


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

/*
template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(dense_tensor_i<N + M, double> &blk,
	const index<N + M> &i) {

	static const char *method =
		"compute_block(dense_tensor_i<N + M, double>&, const index<N + M>&)";

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
		contract_block(isch->second->first, aic.get_index(), ca, cb,
			blk, trc0, true, 1.0);

	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}
*/

template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::compute_block(bool zero,
    dense_tensor_i<N + M, double> &blk, const index<N + M> &i,
    const transf<N + M, double> &tr, double c, cpu_pool &cpus) {

	static const char *method = "compute_block(bool, tensor_i<N + M, double>&, "
		"const index<N + M>&, const transf<N + M, double>&, double, cpu_pool&)";

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

		contract_block(isch->second->first, aic.get_index(), ca, cb,
			blk, tr, zero, c, cpus);
	} catch(...) {
		btod_contract2<N, M, K>::stop_timer();
		throw;
	}

	btod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule() {

	btod_contract2<N, M, K>::start_timer("make_schedule");
	btod_contract2<N, M, K>::start_timer("prepare_sch");

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	ca.req_sync_on();
	cb.req_sync_on();

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());

	std::vector<make_schedule_task*> tasklist;
	task_batch tb;
	libvmm::mutex sch_lock;

	typename orbit_list<k_ordera, double>::iterator ioa1 = ola.begin(),
		ioa2 = ola.begin();
	size_t n = 0, nmax = ola.get_size() / 64;
	if(nmax < 1024) nmax = 1024;
	for(; ioa2 != ola.end(); ioa2++, n++) {

		if(n == nmax) {
			make_schedule_task *t = new make_schedule_task(m_contr,
				m_bta, m_btb, get_symmetry(), m_bidimsc, ola,
				ioa1, ioa2, m_contr_sch, m_sch, sch_lock);
			tasklist.push_back(t);
			tb.push(*t);
			n = 0;
			ioa1 = ioa2;
		}
	}
	if(ioa1 != ola.end()) {
		make_schedule_task *t = new make_schedule_task(m_contr, m_bta,
			m_btb, get_symmetry(), m_bidimsc, ola, ioa1, ioa2,
			m_contr_sch, m_sch, sch_lock);
		tasklist.push_back(t);
		tb.push(*t);
	}

	btod_contract2<N, M, K>::stop_timer("prepare_sch");

	try {
		tb.wait();
	} catch(...) {
		for(size_t i = 0; i < tasklist.size(); i++) delete tasklist[i];
		btod_contract2<N, M, K>::stop_timer("make_schedule");
		throw;
	}

	for(size_t i = 0; i < tasklist.size(); i++) delete tasklist[i];

	ca.req_sync_off();
	cb.req_sync_off();

	for(typename schedule_t::iterator i = m_contr_sch.begin();
		i != m_contr_sch.end(); i++) {
		m_sch.insert(i->first);
	}

	btod_contract2<N, M, K>::stop_timer("make_schedule");
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
btod_contract2<N, M, K>::make_schedule_task::make_schedule_task(
	const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb,
	const symmetry<k_orderc, double> &symc,
	const dimensions<k_orderc> &bidimsc,
	const orbit_list<k_ordera, double> &ola,
	const typename orbit_list<k_ordera, double>::iterator &ioa1,
	const typename orbit_list<k_ordera, double>::iterator &ioa2,
	schedule_t &contr_sch, assignment_schedule<k_orderc, double> &sch,
	libvmm::mutex &sch_lock) :

	m_contr(contr),
	m_ca(bta), m_bidimsa(bta.get_bis().get_block_index_dims()),
	m_cb(btb), m_bidimsb(btb.get_bis().get_block_index_dims()),
	m_symc(symc), m_bidimsc(bidimsc),
	m_ola(ola), m_ioa1(ioa1), m_ioa2(ioa2),
	m_contr_sch(contr_sch), m_sch(sch), m_sch_lock(sch_lock) {

}


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::make_schedule_task::k_clazz =
	"btod_contract2<N, M, K>::make_schedule_task";


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::perform(cpu_pool &cpus) throw(exception) {

    auto_cpu_lock cpu(cpus);

	make_schedule_task::start_timer("local");

	orbit_list<k_orderc, double> olc(m_symc);

	for(typename orbit_list<k_ordera, double>::iterator ioa = m_ioa1;
		ioa != m_ioa2; ioa++) {

		orbit<k_ordera, double> oa(m_ca.req_const_symmetry(),
			m_ola.get_index(ioa));
		if(!oa.is_allowed()) continue;
		if(m_ca.req_is_zero_block(m_ola.get_index(ioa))) continue;

		abs_index<k_ordera> acia(m_ola.get_index(ioa), m_bidimsa);
		for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
			ia != oa.end(); ia++) {

			abs_index<k_ordera> aia(oa.get_abs_index(ia),
				m_bidimsa);
			make_schedule_a(olc, aia, acia, oa.get_transf(ia));
		}
	}

	make_schedule_task::stop_timer("local");

	make_schedule_task::start_timer("merge");
	merge_schedule();
	make_schedule_task::stop_timer("merge");
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::make_schedule_a(
	const orbit_list<k_orderc, double> &olc,
	const abs_index<k_ordera> &aia, const abs_index<k_ordera> &acia,
	const transf<k_ordera, double> &tra) {

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
	for(size_t i = 0; i < M; i++) iu2[i] = m_bidimsb[useqb[i]] - 1;
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
		abs_index<k_orderc> aic(ic, m_bidimsc);
		if(!olc.contains(aic.get_abs_index())) continue;

		// Construct the index in B
		index<k_orderb> ib;
		for(size_t i = 0; i < k_orderb; i++) {
			register size_t k = conn[k_orderc + k_ordera + i];
			if(k < k_orderc) ib[i] = ic[k];
			else ib[i] = ia[k - k_orderc];
		}
		make_schedule_b(acia, tra, ib, aic);
	} while(aiu.inc());
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::make_schedule_b(
	const abs_index<k_ordera> &acia, const transf<k_ordera, double> &tra,
	const index<k_orderb> &ib, const abs_index<k_orderc> &acic) {

	orbit<k_orderb, double> ob(m_cb.req_const_symmetry(), ib);
	if(!ob.is_allowed()) return;

	abs_index<k_orderb> acib(ob.get_abs_canonical_index(), m_bidimsb);
	if(m_cb.req_is_zero_block(acib.get_index())) return;

	const transf<k_orderb, double> &trb = ob.get_transf(ib);
	block_contr_t bc(acia.get_abs_index(), acib.get_abs_index(),
		tra.get_coeff() * trb.get_coeff(),
		permutation<N + K>(tra.get_perm(), true),
		permutation<M + K>(trb.get_perm(), true));
	schedule_block_contraction(acic, bc);
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::schedule_block_contraction(
	const abs_index<k_orderc> &acic, const block_contr_t &bc) {

	typename schedule_t::iterator isch;
	block_contr_list_pair_t *lstpair = 0;

	std::pair<typename schedule_t::iterator, bool> r =
		m_contr_sch_local.insert(
			std::pair<size_t, block_contr_list_pair_t*>(
			acic.get_abs_index(), lstpair));
	// Check whether this is the first contraction for this block in C
	if(r.second) {
		lstpair = new block_contr_list_pair_t;
		lstpair->first.push_back(bc);
		lstpair->second = false;
		r.first->second = lstpair;
		return;
	} else {
		isch = r.first;
	}

	lstpair = isch->second;
	merge_node(bc, lstpair->first, lstpair->first.begin());
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::merge_schedule() {

	libvmm::auto_lock lock(m_sch_lock);

	for(typename schedule_t::iterator isrc = m_contr_sch_local.begin();
		isrc != m_contr_sch_local.end(); isrc++) {

		std::pair<typename schedule_t::iterator, bool> rdst =
			m_contr_sch.insert(*isrc);
		if(!rdst.second) {
			typename schedule_t::iterator idst = rdst.first;
			merge_lists(isrc->second->first, idst->second->first);
			delete isrc->second;
		}
		isrc->second = 0;
	}
	m_contr_sch_local.clear();
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_schedule_task::merge_lists(
	const block_contr_list_t &src, block_contr_list_t &dst) {

	typename block_contr_list_t::const_iterator isrc = src.begin();
	typename block_contr_list_t::iterator idst = dst.begin();
	for(; isrc != src.end(); isrc++) {
		idst = merge_node(*isrc, dst, idst);
	}
}


template<size_t N, size_t M, size_t K>
typename btod_contract2<N, M, K>::block_contr_list_t::iterator
btod_contract2<N, M, K>::make_schedule_task::merge_node(
	const block_contr_t &bc, block_contr_list_t &lst,
	const typename block_contr_list_t::iterator &begin) {

	typename block_contr_list_t::iterator ilst = begin;

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

	return ilst;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::contract_block(
	block_contr_list_t &lst, const index<k_orderc> &idxc,
	block_tensor_ctrl<k_ordera, double> &ca,
	block_tensor_ctrl<k_orderb, double> &cb,
	dense_tensor_i<k_orderc, double> &tc, const transf<k_orderc, double> &trc,
	bool zero, double c, cpu_pool &cpus) {

	if(zero) tod_set<k_orderc>().perform(cpus, tc);

	std::list< index<k_ordera> > blksa;
	std::list< index<k_orderb> > blksb;

	for(typename block_contr_list_t::iterator ilst = lst.begin();
		ilst != lst.end(); ilst++) {

		abs_index<k_ordera> aia(ilst->m_absidxa, m_bidimsa);
		abs_index<k_orderb> aib(ilst->m_absidxb, m_bidimsb);
		const index<k_ordera> &ia = aia.get_index();
		const index<k_orderb> &ib = aib.get_index();

		bool zeroa = ca.req_is_zero_block(ia);
		bool zerob = cb.req_is_zero_block(ib);
		if(zeroa || zerob) continue;

		dense_tensor_i<k_ordera, double> &blka = ca.req_block(ia);
		dense_tensor_i<k_orderb, double> &blkb = cb.req_block(ib);
		blksa.push_back(ia);
		blksb.push_back(ib);

		contraction2<N, M, K> contr(m_contr);
		contr.permute_a(ilst->m_perma);
		contr.permute_b(ilst->m_permb);
		contr.permute_c(trc.get_perm());

        double kc = ilst->m_c * trc.get_coeff() * c;
		tod_contract2<N, M, K>(contr, blka, blkb).
		    perform(cpus, false, kc, tc);
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
	sequence< K, mask<2 * (N + K)> > msks;
	for (size_t i = 0, k = 0; i < 2 * (N + K); i++) {
		seq1[i] = i;
		if (conn[i + 2 * N] < 2 * N) { // remaining indexes
			seq2[conn[i + 2 * N]] = i;
		}
		else if (i < N + K) { // contracted indexes
			size_t j = 2 * (N + k);
			msks[k][j] = true;
			seq2[j] = i;
			j++;
			msks[k][j] = true;
			seq2[j] = conn[i + 2 * N] - 2 * N;
			k++;
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

	so_stabilize<2 * (N + K), 2 * K, K, double> so_stab(xsymab);
	for (size_t k = 0; k < K; k++) so_stab.add_mask(msks[k]);
	so_stab.perform(
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
	xbisab.match_splits();

	return xbisab;
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M>
btod_contract2_symmetry_builder_base<N, M, K>::make_bis(
	const contraction2<N, M, K> &contr,
	const block_index_space<N + M + 2 * K> &xbis) {

	const dimensions<N + M + 2 * K> &xdims = xbis.get_dims();
	const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();
	mask<N + M + 2 * K> msk;
	size_t seq1[N + M], seq2[N + M];
	for (size_t i = 0, j = 0; i < N + M + 2 * K; i++) {
		if (conn[N + M + i] < N + M) {
			msk[i] = true;
			seq1[j] = j;
			seq2[j] = conn[N + M + i];
			j++;
		}
	}
	block_index_subspace_builder<N + M, 2 * K> rbb(xbis, msk);
	permutation_builder<N + M> pb(seq1, seq2);
	block_index_space<N + M> bis = rbb.get_bis();
	bis.permute(pb.get_perm());

	return bis;
}


template<size_t N, size_t M, size_t K>
void btod_contract2_symmetry_builder_base<N, M, K>::make_symmetry(
	const contraction2<N, M, K> &contr,
	block_tensor_i<N + K, double> &bta,
	block_tensor_i<M + K, double> &btb) {

	block_tensor_ctrl<N + K, double> ca(bta);
	block_tensor_ctrl<M + K, double> cb(btb);
	block_index_space<N + M + 2 * K> xbis(m_xbis);

	const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

	sequence<N + M + 2 * K, size_t> seq1(0), seq2(0);
	sequence< K, mask<N + M + 2 * K> > msks;
	for (size_t i = 0, k = 0; i < N + M + 2 * K; i++) {
		seq1[i] = i;
		if (conn[i + N + M] < N + M) { // remaining indexes
			seq2[conn[i + N + M]] = i;
		}
		else if (i < N + K) { // contracted indexes
			size_t j = N + M + 2 * k;
			msks[k][j] = true;
			seq2[j] = i;
			j++;
			msks[k][j] = true;
			seq2[j] = conn[i + N + M] - (N + M);
			k++;
		}
	}
	permutation_builder<N + M + 2 * K> pb(seq2, seq1);
	xbis.permute(pb.get_perm());
	symmetry<N + M + 2 * K, double> xsymab(xbis);

	so_concat<N + K, M + K, double>(ca.req_const_symmetry(),
			cb.req_const_symmetry(), pb.get_perm()).perform(xsymab);

	so_stabilize<N + M + 2 * K, 2 * K, K, double> so_stab(xsymab);
	for (size_t k = 0; k < K; k++) so_stab.add_mask(msks[k]);
	so_stab.perform(btod_contract2_symmetry_builder_base<N, M, K>::
		get_symmetry());

}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_IMPL_H
