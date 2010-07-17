#ifndef LIBTENSOR_BTOD_DIRSUM_H
#define LIBTENSOR_BTOD_DIRSUM_H

#include <list>
#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
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
#include "../symmetry/so_proj_up.h"
#include "../symmetry/so_union.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Computes the direct sum of two block tensors
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.

	Given two tensors \f$ a_{ij\cdots} \f$ and \f$ b_{mn\cdots} \f$,
	the operation computes
	\f$ c_{ij\cdots mn\cdots} = k_a a_{ij\cdots} + k_b b_{mn\cdots} \f$.

	The order of %tensor indexes in the result can be specified using
	a permutation.

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_dirsum :
	public additive_btod<N + M>,
	public timings< btod_dirsum<N, M> > {

public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	struct schrec {
		size_t absidxa, absidxb;
		bool zeroa, zerob;
		double ka, kb;
		permutation<k_orderc> permc;
	};
	typedef std::map<size_t, schrec> schedule_t;

private:
	block_tensor_i<k_ordera, double> &m_bta; //!< First %tensor (A)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second %tensor (B)
	double m_ka; //!< Coefficient A
	double m_kb; //!< Coefficient B
	permutation<k_orderc> m_permc; //!< Permutation of the result
	block_index_space<k_orderc>
		m_bisc; //!< Block index space of the result
	symmetry<k_orderc, double> m_sym; //!< Symmetry of the result
	dimensions<k_ordera> m_bidimsa; //!< Block %index dims of A
	dimensions<k_orderb> m_bidimsb; //!< Block %index dims of B
	dimensions<k_orderc> m_bidimsc; //!< Block %index dims of the result
	schedule_t m_op_sch; //!< Direct sum schedule
	assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
	/**	\brief Initializes the operation
	 **/
	btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb);

	/**	\brief Initializes the operation
	 **/
	btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb,
		const permutation<k_orderc> &permc);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_dirsum() { }

	virtual const block_index_space<N + M> &get_bis() const {
		return m_bisc;
	}

	virtual const symmetry<N + M, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N + M, double> &get_schedule() const {
		return m_sch;
	}

	virtual void sync_on();
	virtual void sync_off();

	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i);

	virtual void compute_block(tensor_i<N + M, double> &blk,
		const index<N + M> &i, const transf<N + M, double> &tr,
		double c);

	using additive_btod<N + M>::perform;

private:
	static block_index_space<N + M> make_bis(
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb,
		const permutation<k_orderc> &permc);
	void make_symmetry();
	void make_schedule();
	void make_schedule(const orbit<k_ordera, double> &oa, bool zeroa,
		const orbit<k_orderb, double> &ob, bool zerob,
		const orbit_list<k_orderc, double> &olc);

	void compute_block(tensor_i<N + M, double> &blkc,
		const schrec &rec, const transf<N + M, double> &trc,
		bool zeroc, double kc);

	void do_block_dirsum(block_tensor_ctrl<k_ordera, double> &ctrla,
		block_tensor_ctrl<k_orderb, double> &ctrlb,
		tensor_i<k_orderc, double> &blkc, double kc,
		const index<k_ordera> &ia, double ka,
		const index<k_orderb> &ib, double kb,
		const permutation<k_orderc> &permc, bool zero);

	void do_block_scatter_a(block_tensor_ctrl<k_ordera, double> &ctrla,
		tensor_i<k_orderc, double> &blkc, double kc,
		const index<k_ordera> &ia, double ka,
		const permutation<k_orderc> permc, bool zero);

	void do_block_scatter_b(block_tensor_ctrl<k_orderb, double> &ctrlb,
		tensor_i<k_orderc, double> &blkc, double kc,
		const index<k_orderb> &ib, double kb,
		const permutation<k_orderc> permc, bool zero);

};


template<size_t N, size_t M>
const char *btod_dirsum<N, M>::k_clazz = "btod_dirsum<N, M>";


template<size_t N, size_t M>
btod_dirsum<N, M>::btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
	block_tensor_i<k_orderb, double> &btb, double kb) :

	m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb),
	m_bisc(make_bis(m_bta, m_btb, m_permc)), m_sym(m_bisc),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_bisc.get_block_index_dims()), m_sch(m_bidimsc) {

	make_symmetry();
	make_schedule();
}


template<size_t N, size_t M>
btod_dirsum<N, M>::btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
	block_tensor_i<k_orderb, double> &btb, double kb,
	const permutation<k_orderc> &permc) :

	m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb), m_permc(permc),
	m_bisc(make_bis(m_bta, m_btb, m_permc)), m_sym(m_bisc),
	m_bidimsa(m_bta.get_bis().get_block_index_dims()),
	m_bidimsb(m_btb.get_bis().get_block_index_dims()),
	m_bidimsc(m_bisc.get_block_index_dims()), m_sch(m_bidimsc) {

	make_symmetry();
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
block_index_space<N + M> btod_dirsum<N, M>::make_bis(
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb,
	const permutation<k_orderc> &permc) {

	const block_index_space<k_ordera> &bisa = bta.get_bis();
	const dimensions<k_ordera> &dimsa = bisa.get_dims();
	const block_index_space<k_orderb> &bisb = btb.get_bis();
	const dimensions<k_orderb> &dimsb = bisb.get_dims();

	index<k_orderc> i1, i2;
	for(register size_t i = 0; i < k_ordera; i++)
		i2[i] = dimsa[i] - 1;
	for(register size_t i = 0; i < k_orderb; i++)
		i2[k_ordera + i] = dimsb[i] - 1;

	dimensions<k_orderc> dimsc(index_range<k_orderc>(i1, i2));
	block_index_space<k_orderc> bisc(dimsc);

	mask<k_ordera> mska, mska1;
	mask<k_orderb> mskb, mskb1;
	mask<k_orderc> mskc;
	bool done;
	size_t i;

	i = 0;
	done = false;
	while(!done) {
		while(i < k_ordera && mska[i]) i++;
		if(i == k_ordera) {
			done = true;
			continue;
		}

		size_t typ = bisa.get_type(i);
		for(size_t j = 0; j < k_ordera; j++) {
			mskc[j] = mska1[j] = bisa.get_type(j) == typ;
		}
		const split_points &pts = bisa.get_splits(typ);
		for(size_t j = 0; j < pts.get_num_points(); j++)
			bisc.split(mskc, pts[j]);

		mska |= mska1;
	}
	for(size_t j = 0; j < k_ordera; j++) mskc[j] = false;

	i = 0;
	done = false;
	while(!done) {
		while(i < k_orderb && mskb[i]) i++;
		if(i == k_orderb) {
			done = true;
			continue;
		}

		size_t typ = bisb.get_type(i);
		for(size_t j = 0; j < k_orderb; j++) {
			mskc[k_ordera + j] = mskb1[j] =
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
void btod_dirsum<N, M>::make_symmetry() {

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	size_t seq[k_orderc];
	for(size_t i = 0; i < k_orderc; i++) seq[i] = i;
	m_permc.apply(seq);

	mask<k_orderc> xma, xmb;
	sequence<N, size_t> xseqa1(0), xseqa2(0);
	sequence<M, size_t> xseqb1(0), xseqb2(0);

	for(size_t i = 0, ja = 0, jb = 0; i < k_orderc; i++) {
		if(seq[i] < k_ordera) {
			xma[i] = true;
			xseqa1[ja] = ja;
			xseqa2[ja] = seq[i];
			ja++;
		} else {
			xmb[i] = true;
			xseqb1[jb] = jb;
			xseqb2[jb] = seq[i] - k_ordera;
			jb++;
		}
	}

	permutation_builder<N> xpba(xseqa2, xseqa1);
	permutation_builder<M> xpbb(xseqb2, xseqb1);
	symmetry<k_orderc, double> xsyma(m_bisc);
	symmetry<k_orderc, double> xsymb(m_bisc);
	so_proj_up<N, M, double>(ca.req_const_symmetry(), xpba.get_perm(), xma).
		perform(xsyma);
	so_proj_up<M, N, double>(cb.req_const_symmetry(), xpbb.get_perm(), xmb).
		perform(xsymb);
	so_union<k_orderc, double>(xsyma, xsymb).perform(m_sym);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::make_schedule() {

	btod_dirsum<N, M>::start_timer("make_schedule");

	block_tensor_ctrl<k_ordera, double> ca(m_bta);
	block_tensor_ctrl<k_orderb, double> cb(m_btb);

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	orbit_list<k_orderb, double> olb(cb.req_const_symmetry());
	orbit_list<k_orderc, double> olc(m_sym);

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


} // namespace libtensor

#endif // LIBTENOSR_BTOD_DIRSUM_H
