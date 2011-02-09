#ifndef LIBTENSOR_BTOD_DIRSUM_IMPL_H
#define LIBTENSOR_BTOD_DIRSUM_IMPL_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../core/mask.h"
#include "../tod/tod_dirsum.h"
#include "../tod/tod_scale.h"
#include "../tod/tod_scatter.h"
#include "../tod/tod_set.h"
#include "../symmetry/so_concat.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "transf_double.h"

namespace libtensor {

template<size_t N, size_t M>
const char *btod_dirsum_clazz<N, M>::k_clazz = "btod_dirsum<N, M>";

template<size_t N, size_t M>
const char *btod_dirsum<N, M>::k_clazz =
		btod_dirsum_clazz<N, M>::k_clazz;

template<size_t N, size_t M>
btod_dirsum<N, M>::btod_dirsum(
		block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb) :

	m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb),
	m_sym_bld(m_bta, m_ka, m_btb, m_kb, m_permc),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_sym_bld.get_bis().get_block_index_dims()), m_sch(m_bidimsc) {

	make_schedule();
}


template<size_t N, size_t M>
btod_dirsum<N, M>::btod_dirsum(
		block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb,
		const permutation<k_orderc> &permc) :

	m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb), m_permc(permc),
	m_sym_bld(m_bta, m_ka, m_btb, m_kb, m_permc),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_sym_bld.get_bis().get_block_index_dims()), m_sch(m_bidimsc) {

	make_schedule();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::sync_on() {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	ctrla.req_sync_on();
	ctrlb.req_sync_on();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::sync_off() {

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	ctrla.req_sync_off();
	ctrlb.req_sync_off();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::compute_block(tensor_i<N + M, double> &blkc,
	const index<N + M> &ic) {

	static const char *method =
		"compute_block(tensor_i<N + M, double>&, const index<N + M>&)";

	btod_dirsum<N, M>::start_timer();

	try {

		abs_index<k_orderc> aic(ic, m_bidimsc);
		typename schedule_t::const_iterator isch =
			m_op_sch.find(aic.get_abs_index());
		if(isch == m_op_sch.end()) {
			tod_set<k_orderc>().perform(blkc);
		} else {
			transf<k_orderc, double> trc0;
			compute_block(blkc, isch->second, trc0, true, 1.0);
		}

	} catch(...) {
		btod_dirsum<N, M>::stop_timer();
		throw;
	}

	btod_dirsum<N, M>::stop_timer();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::compute_block(tensor_i<N + M, double> &blkc,
	const index<N + M> &ic, const transf<N + M, double> &trc,
	double kc) {

	static const char *method = "compute_block(tensor_i<N + M, double>&, "
		"const index<N + M>&, const transf<N + M, double>&, double)";

	btod_dirsum<N, M>::start_timer();

	try {

		abs_index<k_orderc> aic(ic, m_bidimsc);
		typename schedule_t::const_iterator isch =
			m_op_sch.find(aic.get_abs_index());
		if(isch != m_op_sch.end()) {
			compute_block(blkc, isch->second, trc, false, kc);
		}

	} catch(...) {
		btod_dirsum<N, M>::stop_timer();
		throw;
	}

	btod_dirsum<N, M>::stop_timer();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::compute_block(tensor_i<N + M, double> &blkc,
	const schrec &rec, const transf<N + M, double> &trc, bool zeroc,
	double kc) {

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	abs_index<k_ordera> aia(rec.absidxa, m_bidimsa);
	abs_index<k_orderb> aib(rec.absidxb, m_bidimsb);
	double kc1 = kc * trc.get_coeff();
	permutation<k_orderc> permc1(rec.permc); permc1.permute(trc.get_perm());
	if(rec.zerob) {
		permutation<k_orderc> cycc;
		for(size_t i = 0; i < k_orderc - 1; i++) cycc.permute(i, i + 1);
		permutation<k_orderc> permc2;
		for(size_t i = 0; i < k_ordera; i++) permc2.permute(cycc);
		permc2.permute(permc1);
		do_block_scatter_a(ca, blkc, kc1, aia.get_index(), rec.ka,
			permc2, zeroc);
	} else if(rec.zeroa) {
		do_block_scatter_b(cb, blkc, kc1, aib.get_index(), rec.kb,
			permc1, zeroc);
	} else {
		do_block_dirsum(ca, cb, blkc, kc1, aia.get_index(), rec.ka,
			aib.get_index(), rec.kb, permc1, zeroc);
	}
}

template<size_t N, size_t M>
void btod_dirsum<N, M>::make_schedule() {

	btod_dirsum<N, M>::start_timer("make_schedule");

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);
	const symmetry<k_orderc, double> &sym = m_sym_bld.get_symmetry();

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	orbit_list<k_orderb, double> olb(cb.req_const_symmetry());
	orbit_list<k_orderc, double> olc(sym);

	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		bool zeroa = ca.req_is_zero_block(ola.get_index(ioa));

		orbit<k_ordera, double> oa(ca.req_const_symmetry(),
			ola.get_index(ioa));

		for(typename orbit_list<k_orderb, double>::iterator iob =
			olb.begin(); iob != olb.end(); iob++) {

			bool zerob = cb.req_is_zero_block(olb.get_index(iob));
			if(zeroa && zerob) continue;

			orbit<k_orderb, double> ob(cb.req_const_symmetry(),
				olb.get_index(iob));

				make_schedule(oa, zeroa, ob, zerob, olc);
		}
	}

	// check if there are orbits in olc which have not been added to schedule
	permutation<k_orderc> pinvc(m_permc, true);
	for (typename orbit_list<k_orderc, double>::iterator ioc = olc.begin();
			ioc != olc.end(); ioc++) {

		if (m_sch.contains(olc.get_abs_index(ioc))) continue;

		// identify index of A and index of B
		index<k_orderc> idxc(olc.get_index(ioc));
		idxc.permute(pinvc);
		index<k_ordera> idxa;
		for (size_t i = 0; i < k_ordera; i++) idxa[i] = idxc[i];
		index<k_orderb> idxb;
		for (size_t i = 0; i < k_orderb; i++) idxb[i] = idxc[i + k_ordera];

		orbit<k_ordera, double> oa(ca.req_const_symmetry(), idxa);
		orbit<k_orderb, double> ob(cb.req_const_symmetry(), idxb);

		bool zeroa = ! oa.is_allowed();
		bool zerob = ! ob.is_allowed();

		if (zeroa == zerob) continue;

		if (! zeroa) {
			abs_index<k_ordera> ai(oa.get_abs_canonical_index(),
					m_bta.get_bis().get_block_index_dims());
			zeroa = ca.req_is_zero_block(ai.get_index());
		}


		if (! zerob) {
			abs_index<k_orderb> bi(ob.get_abs_canonical_index(),
					m_btb.get_bis().get_block_index_dims());
			zerob = cb.req_is_zero_block(bi.get_index());
		}

		make_schedule(oa, zeroa, ob, zerob, olc);
	}

	btod_dirsum<N, M>::stop_timer("make_schedule");
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::make_schedule(const orbit<k_ordera, double> &oa,
	bool zeroa, const orbit<k_orderb, double> &ob, bool zerob,
	const orbit_list<k_orderc, double> &olc) {

	size_t seqa[k_ordera];
	size_t seqb[k_orderb];
	size_t seqc1[k_orderc], seqc2[k_orderc];

	for(size_t i = 0; i < k_orderc; i++) seqc1[i] = i;

	for(typename orbit<k_ordera, double>::iterator ia = oa.begin();
		ia != oa.end(); ia++) {

		abs_index<k_ordera> aidxa(oa.get_abs_index(ia), m_bidimsa);
		const index<k_ordera> &idxa = aidxa.get_index();
		const transf<k_ordera, double> &tra = oa.get_transf(
				aidxa.get_abs_index());

		for(size_t i = 0; i < k_ordera; i++) seqa[i] = i;
		tra.get_perm().apply(seqa);

		for(typename orbit<k_orderb, double>::iterator ib = ob.begin();
				ib != ob.end(); ib++) {

			abs_index<k_orderb> aidxb(ob.get_abs_index(ib), m_bidimsb);
			const index<k_orderb> &idxb = aidxb.get_index();
			const transf<k_orderb, double> &trb = ob.get_transf(
					aidxb.get_abs_index());

			for(size_t i = 0; i < k_orderb; i++) seqb[i] = i;
			trb.get_perm().apply(seqb);

			index<k_orderc> idxc;
			for(size_t i = 0; i < k_ordera; i++) {
				idxc[i] = idxa[i];
				seqc2[i] = seqa[i];
			}
			for(size_t i = 0; i < k_orderb; i++) {
				idxc[k_ordera + i] = idxb[i];
				seqc2[k_ordera + i] = k_ordera + seqb[i];
			}

			idxc.permute(m_permc);
			m_permc.apply(seqc2);

			abs_index<k_orderc> aidxc(idxc, m_bidimsc);
			if(!olc.contains(aidxc.get_abs_index())) continue;

			permutation_builder<k_orderc> pbc(seqc2, seqc1);
			schrec rec;
			rec.absidxa = aidxa.get_abs_index();
			rec.absidxb = aidxb.get_abs_index();
			rec.zeroa = zeroa;
			rec.zerob = zerob;
			rec.ka = m_ka * tra.get_coeff();
			rec.kb = m_kb * trb.get_coeff();
			rec.permc.permute(pbc.get_perm());
			m_op_sch.insert(std::pair<size_t, schrec>(
					aidxc.get_abs_index(), rec));
			m_sch.insert(aidxc.get_abs_index());

		} // for ib

	} // for ia
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_dirsum(
	block_tensor_ctrl<k_ordera, double> &ctrla,
	block_tensor_ctrl<k_orderb, double> &ctrlb,
	tensor_i<k_orderc, double> &blkc, double kc,
	const index<k_ordera> &ia, double ka,
	const index<k_orderb> &ib, double kb,
	const permutation<k_orderc> &permc, bool zero) {

	tensor_i<k_ordera, double> &blka = ctrla.req_block(ia);
	tensor_i<k_orderb, double> &blkb = ctrlb.req_block(ib);

	if(zero) {
		tod_dirsum<N, M>(blka, ka, blkb, kb, permc).perform(blkc);
		if(kc != 1.0) {
			tod_scale<k_orderc>(blkc, kc).perform();
		}
	} else {
		tod_dirsum<N, M>(blka, ka, blkb, kb, permc).perform(blkc, kc);
	}

	ctrla.ret_block(ia);
	ctrlb.ret_block(ib);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_scatter_a(
	block_tensor_ctrl<k_ordera, double> &ctrla,
	tensor_i<k_orderc, double> &blkc, double kc,
	const index<k_ordera> &ia, double ka,
	const permutation<k_orderc> permc, bool zero) {

	tensor_i<k_ordera, double> &blka = ctrla.req_block(ia);

	if(zero) {
		tod_scatter<N, M>(blka, ka, permc).perform(blkc);
		if(kc != 1.0) {
			tod_scale<k_orderc>(blkc, kc).perform();
		}
	} else {
		tod_scatter<N, M>(blka, ka, permc).perform(blkc, kc);
	}

	ctrla.ret_block(ia);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_scatter_b(
	block_tensor_ctrl<k_orderb, double> &ctrlb,
	tensor_i<k_orderc, double> &blkc, double kc,
	const index<k_orderb> &ib, double kb,
	const permutation<k_orderc> permc, bool zero) {

	tensor_i<k_orderb, double> &blkb = ctrlb.req_block(ib);

	if(zero) {
		tod_scatter<M, N>(blkb, kb, permc).perform(blkc);
		if(kc != 1.0) {
			tod_scale<k_orderc>(blkc, kc).perform();
		}
	} else {
		tod_scatter<M, N>(blkb, kb, permc).perform(blkc, kc);
	}

	ctrlb.ret_block(ib);
}

template<size_t N, size_t M>
block_index_space<N + M>
btod_dirsum_symmetry_builder_base<N, M>::make_bis(
	const block_index_space<N> &bisa,
	const block_index_space<M> &bisb,
	const permutation<N + M> &permc) {

	const dimensions<N> &dimsa = bisa.get_dims();
	const dimensions<M> &dimsb = bisb.get_dims();

	index<N + M> i1, i2;
	for(register size_t i = 0; i < N; i++)
		i2[i] = dimsa[i] - 1;
	for(register size_t i = 0; i < M; i++)
		i2[N + i] = dimsb[i] - 1;

	dimensions<N + M> dimsc(index_range<N + M>(i1, i2));
	block_index_space<N + M> bisc(dimsc);

	mask<N> mska, mska1;
	mask<M> mskb, mskb1;
	mask<N + M> mskc;
	bool done;
	size_t i;

	i = 0;
	done = false;
	while(!done) {
		while(i < N && mska[i]) i++;
		if(i == N) {
			done = true;
			continue;
		}

		size_t typ = bisa.get_type(i);
		for(size_t j = 0; j < N; j++) {
			mskc[j] = mska1[j] = bisa.get_type(j) == typ;
		}
		const split_points &pts = bisa.get_splits(typ);
		for(size_t j = 0; j < pts.get_num_points(); j++)
			bisc.split(mskc, pts[j]);

		mska |= mska1;
	}
	for(size_t j = 0; j < N; j++) mskc[j] = false;

	i = 0;
	done = false;
	while(!done) {
		while(i < M && mskb[i]) i++;
		if(i == M) {
			done = true;
			continue;
		}

		size_t typ = bisb.get_type(i);
		for(size_t j = 0; j < M; j++) {
			mskc[N + j] = mskb1[j] =
				bisb.get_type(j) == typ;
		}
		const split_points &pts = bisb.get_splits(typ);
		for(size_t j = 0; j < pts.get_num_points(); j++)
			bisc.split(mskc, pts[j]);

		mskb |= mskb1;
	}

	bisc.match_splits();
	bisc.permute(permc);

	return bisc;
}

template<size_t N, size_t M>
void btod_dirsum_symmetry_builder_base<N, M>::make_symmetry(
		block_tensor_i<N, double> &bta,
		block_tensor_i<M, double> &btb,
		const permutation<N + M> &permc) {

	block_tensor_ctrl<N, double> ca(bta);
	block_tensor_ctrl<M, double> cb(btb);

	so_concat<N, M, double>(ca.req_const_symmetry(),
			cb.req_const_symmetry(), permc, true).perform(m_sym);
}

template<size_t N, size_t M>
btod_dirsum_symmetry_builder<N, M>::btod_dirsum_symmetry_builder(
		block_tensor_i<N, double> &bta, double ka,
		block_tensor_i<M, double> &btb, double kb,
		const permutation<N + M> &permc) :
		btod_dirsum_symmetry_builder_base<N, M>(
				bta.get_bis(), btb.get_bis(), permc) {

	btod_dirsum_symmetry_builder_base<N, M>::make_symmetry(bta, btb, permc);

}

template<size_t N>
btod_dirsum_symmetry_builder<N, N>::btod_dirsum_symmetry_builder(
		block_tensor_i<N, double> &bta, double ka,
		block_tensor_i<N, double> &btb, double kb,
		const permutation<N + N> &permc) :
		btod_dirsum_symmetry_builder_base<N, N>(bta.get_bis(),
				btb.get_bis(), permc) {

	make_symmetry(bta, ka, btb, kb, permc);

}


template<size_t N>
void btod_dirsum_symmetry_builder<N, N>::make_symmetry(
		block_tensor_i<N, double> &bta, double ka,
		block_tensor_i<N, double> &btb, double kb,
		const permutation<N + N> &permc) {

	btod_dirsum_symmetry_builder_base<N, N>::make_symmetry(bta, btb, permc);

	if ((&bta == &btb) && (ka == kb)) {
		permutation<N + N> perm;
		for (size_t i = 0; i < N; i++) perm.permute(i, i + N);

		sequence<N + N, size_t> seq1(0), seq2(0);
		for (size_t i = 0; i < N; i++) seq1[i] = seq2[i + N] = i;
		for (size_t i = 0; i < N; i++) seq1[i + N] = seq2[i] = i + N;
		seq1.permute(permc);
		seq2.permute(permc);

		permutation_builder<N + N> pb(seq2, seq1);

		se_perm<N + N, double> sp(pb.get_perm(), true);

		btod_dirsum_symmetry_builder_base<N, N>::get_sym().insert(sp);
	}
}



} // namespace libtensor


#endif // LIBTENOSR_BTOD_DIRSUM_IMPL_H
