#ifndef LIBTENSOR_BTOD_DIRSUM_H
#define LIBTENSOR_BTOD_DIRSUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../core/mask.h"
#include "../tod/tod_dirsum.h"
#include "../tod/tod_scale.h"
#include "../tod/tod_scatter.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"
#include "btod_additive.h"
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
	public btod_additive<N + M>,
	public timings< btod_dirsum<N, M> > {

public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	block_tensor_i<k_ordera, double> &m_bta; //!< First %tensor (A)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second %tensor (B)
	double m_ka; //!< Coefficient A
	double m_kb; //!< Coefficient B
	permutation<k_orderc> m_permc; //!< Permutation of the result
	block_index_space<k_orderc>
		m_bisc; //!< Block index space of the result

public:
	/**	\brief Initializes the operation
	 **/
	btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb) :
		m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb),
		m_bisc(mk_bisc(bta, btb)) { }

	/**	\brief Initializes the operation
	 **/
	btod_dirsum(block_tensor_i<k_ordera, double> &bta, double ka,
		block_tensor_i<k_orderb, double> &btb, double kb,
		const permutation<k_orderc> &permc) :
		m_bta(bta), m_btb(btb), m_ka(ka), m_kb(kb), m_permc(permc),
		m_bisc(mk_bisc(bta, btb)) {

		m_bisc.permute(m_permc);
	}

	virtual const block_index_space<N + M> &get_bis() const {
		return m_bisc;
	}

	virtual const symmetry<N + M, double> &get_symmetry() const {
		throw not_implemented(g_ns, k_clazz, "get_symmetry()",
			__FILE__, __LINE__);
	}

	/**	\brief Performs the operation
	 **/
	virtual void perform(block_tensor_i<k_orderc, double> &btc)
		throw(exception);

	/**	\brief Performs the operation (additive)
	 **/
	virtual void perform(block_tensor_i<k_orderc, double> &btc,
		double kc) throw(exception);

	virtual void perform(block_tensor_i<k_orderc, double> &btc,
		const index<k_orderc> &i) throw(exception) {

		throw not_implemented(g_ns, k_clazz, "perform()",
			__FILE__, __LINE__);
	}

private:
	static block_index_space<N + M> mk_bisc(
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);

	void do_perform(block_tensor_i<k_orderc, double> &btc, bool zero,
		double kc);

	void do_block_dirsum(block_tensor_ctrl<k_ordera, double> &ctrla,
		block_tensor_ctrl<k_orderb, double> &ctrlb,
		block_tensor_ctrl<k_orderc, double> &ctrlc,
		const index<k_orderc> &ic, double kc,
		const index<k_ordera> &ia, double ka,
		const index<k_orderb> &ib, double kb,
		const permutation<k_orderc> &permc, bool zero);

	void do_block_scatter_a(block_tensor_ctrl<k_ordera, double> &ctrla,
		block_tensor_ctrl<k_orderc, double> &ctrlc,
		const index<k_orderc> &ic, double kc,
		const index<k_ordera> &ia, double ka,
		const permutation<k_orderc> permc, bool zero);

	void do_block_scatter_b(block_tensor_ctrl<k_orderb, double> &ctrlb,
		block_tensor_ctrl<k_orderc, double> &ctrlc,
		const index<k_orderc> &ic, double kc,
		const index<k_orderb> &ib, double kb,
		const permutation<k_orderc> permc, bool zero);

};


template<size_t N, size_t M>
const char *btod_dirsum<N, M>::k_clazz = "btod_dirsum<N, M>";


template<size_t N, size_t M>
void btod_dirsum<N, M>::perform(block_tensor_i<k_orderc, double> &btc)
	throw(exception) {

	static const char *method = "perform(block_tensor_i<N + M, double>&)";

	if(!m_bisc.equals(btc.get_bis())) {
		throw bad_dimensions(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btc");
	}

	do_perform(btc, true, 1.0);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::perform(block_tensor_i<k_orderc, double> &btc,
	double kc) throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N + M, double>&, double)";

	if(!m_bisc.equals(btc.get_bis())) {
		throw bad_dimensions(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btc");
	}

	do_perform(btc, false, kc);
}


template<size_t N, size_t M>
block_index_space<N + M> btod_dirsum<N, M>::mk_bisc(
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) {

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
		while(i < k_ordera && !mska[i]) i++;
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
		while(i < k_orderb && !mskb[i]) i++;
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

	return bisc;
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_perform(block_tensor_i<k_orderc, double> &btc,
	bool zero, double kc) {

	btod_dirsum<N, M>::start_timer();

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	block_tensor_ctrl<k_orderc, double> ctrlc(btc);

	//	Permutations for tod_scatter
	permutation<k_orderc> perm_cycle;
	permutation<k_orderc> permbc(m_permc);
	permutation<k_orderc> permac;
	{
		size_t seq[k_orderc];
		for(size_t i = 0; i < k_orderc; i++) seq[i] = i;
		m_permc.apply(seq);
		for(size_t i = 0; i < k_orderc - 1; i++)
			perm_cycle.permute(seq[i], seq[i + 1]);
		for(size_t i = 0; i < k_ordera; i++)
			permac.permute(perm_cycle);
	}

	orbit_list<k_ordera, double> ola(ctrla.req_symmetry());
	orbit_list<k_orderb, double> olb(ctrlb.req_symmetry());

	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		index<k_ordera> ia(ola.get_index(ioa));

		for(typename orbit_list<k_orderb, double>::iterator iob =
			olb.begin(); iob != olb.end(); iob++) {

			index<k_orderb> ib(olb.get_index(iob));
			index<k_orderc> ic;

			for(register size_t i = 0; i < k_ordera; i++)
				ic[i] = ia[i];
			for(register size_t i = 0; i < k_orderb; i++)
				ic[k_ordera + i] = ib[i];
			ic.permute(m_permc);

			bool zeroa = ctrla.req_is_zero_block(ia);
			bool zerob = ctrlb.req_is_zero_block(ib);

			if(zero && zeroa && zerob) {
				ctrlc.req_zero_block(ic);
				continue;
			}
			if(zeroa && zerob) {
				continue;
			}

			if(zeroa) {
				do_block_scatter_b(ctrlb, ctrlc, ic, kc, ib,
					m_kb, permbc, zero);
			} else if(zerob) {
				do_block_scatter_a(ctrla, ctrlc, ic, kc, ia,
					m_ka, permac, zero);
			} else {
				do_block_dirsum(ctrla, ctrlb, ctrlc, ic, kc,
					ia, m_ka, ib, m_kb, m_permc, zero);
			}
		}
	}

	btod_dirsum<N, M>::stop_timer();
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_dirsum(
	block_tensor_ctrl<k_ordera, double> &ctrla,
	block_tensor_ctrl<k_orderb, double> &ctrlb,
	block_tensor_ctrl<k_orderc, double> &ctrlc,
	const index<k_orderc> &ic, double kc,
	const index<k_ordera> &ia, double ka,
	const index<k_orderb> &ib, double kb,
	const permutation<k_orderc> &permc, bool zero) {

	tensor_i<k_ordera, double> &blka = ctrla.req_block(ia);
	tensor_i<k_orderb, double> &blkb = ctrlb.req_block(ib);
	tensor_i<k_orderc, double> &blkc = ctrlc.req_block(ic);

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
	ctrlc.ret_block(ic);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_scatter_a(
	block_tensor_ctrl<k_ordera, double> &ctrla,
	block_tensor_ctrl<k_orderc, double> &ctrlc,
	const index<k_orderc> &ic, double kc,
	const index<k_ordera> &ia, double ka,
	const permutation<k_orderc> permc, bool zero) {

	tensor_i<k_ordera, double> &blka = ctrla.req_block(ia);
	tensor_i<k_orderc, double> &blkc = ctrlc.req_block(ic);

	if(zero) {
		tod_scatter<N, M>(blka, ka, permc).perform(blkc);
		if(kc != 1.0) {
			tod_scale<k_orderc>(blkc, kc).perform();
		}
	} else {
		tod_scatter<N, M>(blka, ka, permc).perform(blkc, kc);
	}

	ctrla.ret_block(ia);
	ctrlc.ret_block(ic);
}


template<size_t N, size_t M>
void btod_dirsum<N, M>::do_block_scatter_b(
	block_tensor_ctrl<k_orderb, double> &ctrlb,
	block_tensor_ctrl<k_orderc, double> &ctrlc,
	const index<k_orderc> &ic, double kc,
	const index<k_orderb> &ib, double kb,
	const permutation<k_orderc> permc, bool zero) {

	tensor_i<k_orderb, double> &blkb = ctrlb.req_block(ib);
	tensor_i<k_orderc, double> &blkc = ctrlc.req_block(ic);

	if(zero) {
		tod_scatter<N, M>(blkb, kb, permc).perform(blkc);
		if(kc != 1.0) {
			tod_scale<k_orderc>(blkc, kc).perform();
		}
	} else {
		tod_scatter<N, M>(blkb, kb, permc).perform(blkc, kc);
	}

	ctrlb.ret_block(ib);
	ctrlc.ret_block(ic);
}


} // namespace libtensor

#endif // LIBTENOSR_BTOD_DIRSUM_H