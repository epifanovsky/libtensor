#ifndef LIBTENSOR_BTOD_EWMULT2_IMPL_H
#define LIBTENSOR_BTOD_EWMULT2_IMPL_H

#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_ewmult2.h"
#include "../symmetry/so_concat.h"
#include "bad_block_index_space.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *btod_ewmult2<N, M, K>::k_clazz = "btod_ewmult2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb, double d) :

	m_bta(bta), m_btb(btb), m_d(d),
	m_bisc(make_bisc(bta.get_bis(), permutation<k_ordera>(),
		btb.get_bis(), permutation<k_orderb>(),
		permutation<k_orderc>())),
	m_symc(m_bisc), m_sch(m_bisc.get_block_index_dims()) {

	make_symc();
	make_schedule();
}


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
	const permutation<k_ordera> &perma,
	block_tensor_i<k_orderb, double> &btb,
	const permutation<k_orderb> &permb, const permutation<k_orderc> &permc,
	double d) :

	m_bta(bta), m_perma(perma), m_btb(btb), m_permb(permb), m_permc(permc),
	m_d(d),
	m_bisc(make_bisc(bta.get_bis(), perma, btb.get_bis(), permb, permc)),
	m_symc(m_bisc), m_sch(m_bisc.get_block_index_dims()) {

	make_symc();
	make_schedule();
}


template<size_t N, size_t M, size_t K>
btod_ewmult2<N, M, K>::~btod_ewmult2() {

}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::sync_on() {

	block_tensor_ctrl<k_ordera, double>(m_bta).req_sync_on();
	block_tensor_ctrl<k_orderb, double>(m_btb).req_sync_on();
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::sync_off() {

	block_tensor_ctrl<k_ordera, double>(m_bta).req_sync_off();
	block_tensor_ctrl<k_orderb, double>(m_btb).req_sync_off();
}

/*
template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::compute_block(dense_tensor_i<k_orderc, double> &blk,
	const index<k_orderc> &bidx) {

	transf<k_orderc, double> tr0;
	compute_block_impl(blk, bidx, tr0, true, 1.0);
}*/


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::compute_block(bool zero,
    dense_tensor_i<k_orderc, double> &blk, const index<k_orderc> &bidx,
    const transf<k_orderc, double> &tr, double d, cpu_pool &cpus) {

	compute_block_impl(blk, bidx, tr, zero, d, cpus);
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M + K> btod_ewmult2<N, M, K>::make_bisc(
	const block_index_space<k_ordera> &bisa,
	const permutation<k_ordera> &perma,
	const block_index_space<k_orderb> &bisb,
	const permutation<k_orderb> &permb,
	const permutation<k_orderc> &permc) {

	static const char *method = "make_bisc()";

	//	Block index spaces and dimensions of A and B
	//	in the standard index ordering:
	//	A(ij..pq..) B(mn..pq..)

	block_index_space<k_ordera> bisa1(bisa);
	bisa1.permute(perma);
	block_index_space<k_orderb> bisb1(bisb);
	bisb1.permute(permb);
	dimensions<k_ordera> dimsa1(bisa1.get_dims());
	dimensions<k_orderb> dimsb1(bisb1.get_dims());

	//	Build the dimensions of the result

	index<k_orderc> i1, i2;
	for(size_t i = 0; i < N; i++) i2[i] = dimsa1[i] - 1;
	for(size_t i = 0; i < M; i++) i2[N + i] = dimsb1[i] - 1;
	for(size_t i = 0; i < K; i++) {
		if(dimsa1[N + i] != dimsb1[M + i]) {
			throw bad_block_index_space(g_ns, k_clazz, method,
				__FILE__, __LINE__, "bta,btb");
		}
		if(!bisa1.get_splits(bisa1.get_type(N + i)).equals(
			bisb1.get_splits(bisb1.get_type(M + i)))) {
			throw bad_block_index_space(g_ns, k_clazz, method,
				__FILE__, __LINE__, "bta,btb");
		}
		i2[N + M + i] = dimsa1[N + i] - 1;
	}
	dimensions<k_orderc> dimsc(index_range<k_orderc>(i1, i2));
	block_index_space<k_orderc> bisc(dimsc);

	//	Transfer block index space splits

	mask<k_orderc> mfin, mdone, mtodo;
	for(size_t i = 0; i < k_orderc; i++) mfin[i] = true;
	while(!mdone.equals(mfin)) {
		size_t i;
		for(i = 0; i < k_orderc; i++) mtodo[i] = false;
		for(i = 0; i < k_orderc; i++) if(!mdone[k_orderc - i - 1]) break;
		i = k_orderc - i - 1;
		const split_points *sp = 0;
		if(i < N) {
			size_t j = i;
			size_t typa = bisa1.get_type(j);
			for(size_t k = 0; k < N; k++) {
				mtodo[k] = (bisa1.get_type(k) == typa);
			}
			sp = &bisa1.get_splits(typa);
		} else if(i < N + M) {
			size_t j = i - N;
			size_t typb = bisb1.get_type(j);
			for(size_t k = 0; k < M; k++) {
				mtodo[N + k] = (bisb1.get_type(k) == typb);
			}
			sp = &bisb1.get_splits(typb);
		} else {
			size_t j = i - N - M;
			size_t typa = bisa1.get_type(N + j);
			size_t typb = bisb1.get_type(M + j);
			for(size_t k = 0; k < N; k++) {
				mtodo[k] = (bisa1.get_type(k) == typa);
			}
			for(size_t k = 0; k < M; k++) {
				mtodo[N + k] = (bisb1.get_type(k) == typb);
			}
			for(size_t k = 0; k < K; k++) {
				bool b1 = (bisa1.get_type(N + k) == typa);
				bool b2 = (bisb1.get_type(M + k) == typb);
				if(b1 != b2) {
					throw bad_block_index_space(g_ns,
						k_clazz, method,
						__FILE__, __LINE__, "bta,btb");
				}
				mtodo[N + M + k] = b1;
			}
			sp = &bisa1.get_splits(typa);
		}
		for(size_t j = 0; j < sp->get_num_points(); j++) {
			bisc.split(mtodo, (*sp)[j]);
		}
		mdone |= mtodo;
	}

	bisc.permute(permc);
	return bisc;
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::make_symc() {

/*
	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);

	block_index_space<k_ordera> bisa1(m_bta.get_bis());
	bisa1.permute(m_perma);
	block_index_space<k_orderb> bisb1(m_btb.get_bis());
	bisb1.permute(m_permb);

	dimensions<k_ordera> dimsa1(bisa1.get_dims());
	dimensions<k_orderb> dimsb1(bisb1.get_dims());

	//	Concatenate indexes: form bis for ij..pq..mn..pq..

	index<k_ordera + k_orderb> iab1, iab2;
	for(size_t i = 0; i < k_ordera; i++) iab2[i] = dimsa1[i] - 1;
	for(size_t i = 0; i < k_orderb; i++) iab2[k_ordera + i] = dimsb1[i] - 1;
	dimensions<k_ordera + k_orderb> dimsab(
			index_range<k_ordera + k_orderb>(iab1, iab2));
	block_index_space<k_ordera + k_orderb> bisab(dimsab);

	mask<k_ordera + k_orderb> mdone;
	for(size_t i = 0; i < K; i++) {
		if(mdone[N + i]) continue;
		mask<k_ordera + k_orderb> m;
		size_t typa = bisa1.get_type(N + i);
		for(size_t j = 0; j < k_ordera; j++) {
			m[j] = bisa1.get_type(j) == typa;
		}
		size_t typb = bisb1.get_type(M + i);
		for(size_t j = 0; j < k_orderb; j++) {
			m[k_ordera + j] = bisb1.get_type(j) == typb;
		}
		const split_points &sp = bisa1.get_splits(typa);
		for(size_t j = 0; j < sp.get_num_points(); j++) {
			bisab.split(m, sp[j]);
		}
		mdone |= m;
	}
	for(size_t i = 0; i < N; i++) {
		if(mdone[i]) continue;
		mask<k_ordera + k_orderb> m;
		size_t typa = bisa1.get_type(i);
		for(size_t j = i; j < k_ordera; j++) {
			m[j] = bisa1.get_type(j) == typa;
		}
		const split_points &sp = bisa1.get_splits(typa);
		for(size_t j = 0; j < sp.get_num_points(); j++) {
			bisab.split(m, sp[j]);
		}
		mdone |= m;
	}
	for(size_t i = 0; i < M; i++) {
		if(mdone[k_ordera + i]) continue;
		mask<k_ordera + k_orderb> m;
		size_t typb = bisb1.get_type(i);
		for(size_t j = i; j < k_orderb; j++) {
			m[k_ordera + j] = bisb1.get_type(j) == typb;
		}
		const split_points &sp = bisb1.get_splits(typb);
		for(size_t j = 0; j < sp.get_num_points(); j++) {
			bisab.split(m, sp[j]);
		}
		mdone |= m;
	}

	//	Form symmetry of concatenated indexes

	symmetry<k_ordera + k_orderb, double> symab(bisab);

	sequence<k_ordera, size_t> seqa(0);
	sequence<k_orderb, size_t> seqb(0);
	sequence<k_ordera + k_orderb, size_t> seqab1(0), seqab2(0);
	for(size_t i = 0; i < k_ordera; i++) seqa[i] = i;
	for(size_t i = 0; i < k_orderb; i++) seqb[i] = i;
	for(size_t i = 0; i < k_ordera; i++) {
		seqab1[i] = i;
		seqab2[i] = seqa[i];
	}
	for(size_t i = 0; i < k_orderb; i++) {
		seqab1[k_ordera + i] = k_ordera + i;
		seqab2[k_ordera + i] = k_ordera + seqb[i];
	}
	permutation_builder<k_ordera + k_orderb> permab(seqab2, seqab1);
	so_concat<k_ordera, k_orderb, double>(ctrla.req_const_symmetry(),
		ctrlb.req_const_symmetry(), permab.get_perm()).perform(symab);
*/
/*
	//	Stabilize and remove the extra pq..
	so_stabilize<k_ordera + k_orderb, K, 1, double> stab(symab);
	stab.perform(m_symc);*/
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::make_schedule() {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);

	btod_ewmult2<N, M, K>::start_timer("make_schedule");

	orbit_list<k_orderc, double> ol(m_symc);
	for(typename orbit_list<k_orderc, double>::iterator io = ol.begin();
		io != ol.end(); io++) {

		index<k_ordera> bidxa;
		index<k_orderb> bidxb;
		index<k_orderc> bidxstd(ol.get_index(io));
		bidxstd.permute(permutation<k_orderc>(m_permc, true));
		for(size_t i = 0; i < N; i++) bidxa[i] = bidxstd[i];
		for(size_t i = 0; i < M; i++) bidxb[i] = bidxstd[N + i];
		for(size_t i = 0; i < K; i++) {
			bidxa[N + i] = bidxb[M + i] = bidxstd[N + M +i];
		}
		bidxa.permute(permutation<k_ordera>(m_perma, true));
		bidxb.permute(permutation<k_orderb>(m_permb, true));

		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), bidxa);
		orbit<k_orderb, double> ob(ctrlb.req_const_symmetry(), bidxb);
		if(!oa.is_allowed() || !ob.is_allowed()) continue;

		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
			m_bta.get_bis().get_block_index_dims());
		abs_index<k_orderb> cidxb(ob.get_abs_canonical_index(),
			m_btb.get_bis().get_block_index_dims());
		bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());
		bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());
		if(zeroa || zerob) continue;

		m_sch.insert(ol.get_abs_index(io));
	}

	btod_ewmult2<N, M, K>::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K>
void btod_ewmult2<N, M, K>::compute_block_impl(dense_tensor_i<k_orderc, double> &blk,
	const index<k_orderc> &bidx, const transf<k_orderc, double> &tr,
	bool zero, double d, cpu_pool &cpus) {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);

	btod_ewmult2<N, M, K>::start_timer();

	index<k_ordera> bidxa;
	index<k_orderb> bidxb;
	index<k_orderc> bidxstd(bidx);
	bidxstd.permute(permutation<k_orderc>(m_permc, true));
	for(size_t i = 0; i < N; i++) bidxa[i] = bidxstd[i];
	for(size_t i = 0; i < M; i++) bidxb[i] = bidxstd[N + i];
	for(size_t i = 0; i < K; i++) {
		bidxa[N + i] = bidxb[M + i] = bidxstd[N + M +i];
	}
	bidxa.permute(permutation<k_ordera>(m_perma, true));
	bidxb.permute(permutation<k_orderb>(m_permb, true));

	orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), bidxa);
	orbit<k_orderb, double> ob(ctrlb.req_const_symmetry(), bidxb);

	abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
		m_bta.get_bis().get_block_index_dims());
	const transf<k_ordera, double> &tra = oa.get_transf(bidxa);

	abs_index<k_orderb> cidxb(ob.get_abs_canonical_index(),
		m_btb.get_bis().get_block_index_dims());
	const transf<k_orderb, double> &trb = ob.get_transf(bidxb);

	permutation<k_ordera> perma(tra.get_perm());
	perma.permute(m_perma);
	permutation<k_orderb> permb(trb.get_perm());
	permb.permute(m_permb);
	permutation<k_orderc> permc(m_permc);

	bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());
	bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

	if(zeroa || zerob) {
		btod_ewmult2<N, M, K>::start_timer("zero");
		if(zero) tod_set<k_orderc>().perform(cpus, blk);
		btod_ewmult2<N, M, K>::stop_timer("zero");
		btod_ewmult2<N, M, K>::stop_timer();
		return;
	}

	dense_tensor_i<k_ordera, double> &blka = ctrla.req_block(cidxa.get_index());
	dense_tensor_i<k_orderb, double> &blkb = ctrlb.req_block(cidxb.get_index());

	permc.permute(tr.get_perm());
	double k = m_d * tra.get_coeff() * trb.get_coeff() * tr.get_coeff();
	tod_ewmult2<N, M, K>(blka, perma, blkb, permb, permc, k).
	    perform(cpus, zero, d, blk);

	ctrla.ret_block(cidxa.get_index());
	ctrlb.ret_block(cidxb.get_index());

	btod_ewmult2<N, M, K>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_IMPL_H
