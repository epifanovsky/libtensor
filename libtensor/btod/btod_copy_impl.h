#ifndef LIBTENSOR_BTOD_COPY_IMPL_H
#define LIBTENSOR_BTOD_COPY_IMPL_H

#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_permute.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"

namespace libtensor {


template<size_t N>
const char *btod_copy<N>::k_clazz = "btod_copy<N>";


template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bta, double c) :

	m_bta(bta), m_c(c), m_bis(m_bta.get_bis()),
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	so_copy<N, double>(ctrla.req_const_symmetry()).perform(m_sym);
	make_schedule();
}


template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
	double c) :

	m_bta(bta), m_perm(p), m_c(c), m_bis(mk_bis(m_bta.get_bis(), m_perm)),
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	so_permute<N, double>(ctrla.req_const_symmetry(), m_perm).perform(m_sym);
	make_schedule();
}


template<size_t N>
void btod_copy<N>::sync_on() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	ctrla.req_sync_on();
}


template<size_t N>
void btod_copy<N>::sync_off() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	ctrla.req_sync_off();
}

/*
template<size_t N>
void btod_copy<N>::compute_block(dense_tensor_i<N, double> &blk, const index<N> &ib) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<N> pinv(m_perm, true);

	//	Corresponding index in A
	index<N> ia(ib);
	ia.permute(pinv);

	//	Find the canonical index in A
	orbit<N, double> oa(ctrla.req_const_symmetry(), ia);
	abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);

	//	Transformation for block from canonical A to B
	tensor_transf<N, double> tra(oa.get_tensor_transf(ia));
	tra.permute(m_perm);
	tra.scale(m_c);

	if(!ctrla.req_is_zero_block(acia.get_index())) {
		dense_tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
		tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(blk);
		ctrla.ret_block(acia.get_index());
	} else {
		tod_set<N>().perform(blk);
	}
}
*/

template<size_t N>
void btod_copy<N>::compute_block(bool zero, dense_tensor_i<N, double> &blk,
    const index<N> &ib, const tensor_transf<N, double> &tr,
    double c, cpu_pool &cpus) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<N> pinv(m_perm, true);

	//	Corresponding index in A
	index<N> ia(ib);
	ia.permute(pinv);

	//	Find the canonical index in A
	orbit<N, double> oa(ctrla.req_const_symmetry(), ia);
	abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);

	//	Transformation for block from canonical A to B
	tensor_transf<N, double> tra(oa.get_transf(ia));
	tra.permute(m_perm);
	tra.transform(scalar_transf<double>(m_c));
	tra.transform(tr);

    if(zero) tod_set<N>().perform(cpus, blk);
	if(!ctrla.req_is_zero_block(acia.get_index())) {
		dense_tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
		tod_copy<N>(blka, tra.get_perm(),
		        tra.get_scalar_tr().get_coeff()).perform(cpus, false, c, blk);
		ctrla.ret_block(acia.get_index());
	}
}


template<size_t N>
void btod_copy<N>::make_schedule() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	bool identity = m_perm.is_identity();

	orbit_list<N, double> ola(ctrla.req_const_symmetry());
	for(typename orbit_list<N, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		if(ctrla.req_is_zero_block(ola.get_index(ioa))) continue;

		if(!identity) {
			index<N> bib(ola.get_index(ioa)); bib.permute(m_perm);
			orbit<N, double> ob(m_sym, bib);
			m_sch.insert(ob.get_abs_canonical_index());
		} else {
			m_sch.insert(ola.get_abs_index(ioa));
		}
	}
}


template<size_t N>
block_index_space<N> btod_copy<N>::mk_bis(const block_index_space<N> &bis,
	const permutation<N> &perm) {

	block_index_space<N> bis1(bis);
	bis1.permute(perm);
	return bis1;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_IMPL_H
