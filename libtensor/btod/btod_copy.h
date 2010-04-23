#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <cmath>
#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_add.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_permute.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "../not_implemented.h"

namespace libtensor {


/**	\brief Makes a copy of a block %tensor, applying a permutation and
		a scaling coefficient
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public additive_btod<N>, public timings< btod_copy<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bta; //!< Source block %tensor
	permutation<N> m_perm; //!< Permutation
	double m_c; //!< Scaling coefficient
	block_index_space<N> m_bis; //!< Block %index space of output
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_sym; //!< Symmetry of output
	assignment_schedule<N, double> m_sch;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the copy operation
		\param bt Source block %tensor.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bta, double c = 1.0);

	/**	\brief Initializes the permuted copy operation
		\param bt Source block %tensor.
		\param p Permutation.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_copy() { }
	//@}

	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

//	virtual void perform(block_tensor_i<N, double> &btb) throw(exception);
	using additive_btod<N>::perform;
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx,
		double c) throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual const assignment_schedule<N, double> &get_schedule() {
		return m_sch;
	}
	virtual void perform(block_tensor_i<N, double> &btb, double c)
		throw(exception);
	//@}

protected:
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &ib);
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &ib, double c);

private:
	static block_index_space<N> mk_bis(const block_index_space<N> &bis,
		const permutation<N> &perm);
	void make_schedule();
	void do_perform_nozero(block_tensor_i<N, double> &btb, double c)
		throw(exception);
	void do_perform_zero(block_tensor_i<N, double> &btb, double c)
		throw(exception);
	void do_perform(block_tensor_i<N, double> &btb, const index<N> &dst_idx,
		bool zero, double c) throw(exception);

private:
	btod_copy(const btod_copy<N>&);
	btod_copy<N> &operator=(const btod_copy<N>&);

};


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
	symmetry<N, double> sym1(m_bta.get_bis());
	so_copy<N, double>(ctrla.req_const_symmetry()).perform(sym1);
	so_permute<N, double>(sym1, m_perm).perform(m_sym);
	make_schedule();
}


template<size_t N>
void btod_copy<N>::compute_block(tensor_i<N, double> &blk, const index<N> &ib) {

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
	transf<N, double> tra(oa.get_transf(ia));
	tra.permute(m_perm);
	tra.scale(m_c);

	tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
	tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(blk);
	ctrla.ret_block(acia.get_index());
}


template<size_t N>
void btod_copy<N>::compute_block(tensor_i<N, double> &blk, const index<N> &ib,
	double c) {

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
	transf<N, double> tra(oa.get_transf(ia));
	tra.permute(m_perm);
	tra.scale(m_c);

	tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
	tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(blk, c);
	ctrla.ret_block(acia.get_index());
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

/*
template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}

	btod_copy<N>::start_timer();
	do_perform_zero(btb, 1.0);
	btod_copy<N>::stop_timer();
}
*/

template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb, const index<N> &idx)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, const index<N>&)";

	if(!m_bis.equals(btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}
	if(!m_bidims.contains(idx)) {
		throw out_of_bounds(g_ns, k_clazz, method,
			__FILE__, __LINE__, "idx");
	}

	btod_copy<N>::start_timer();
	do_perform(btb, idx, true, 1.0);
	btod_copy<N>::stop_timer();
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx,
	double c) throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, const index<N>&, double)";

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb, double c)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!m_bis.equals(btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}

	if(fabs(c) == 0.0) return;

	btod_copy<N>::start_timer();
	do_perform_nozero(btb, c);
	btod_copy<N>::stop_timer();
}


template<size_t N>
block_index_space<N> btod_copy<N>::mk_bis(const block_index_space<N> &bis,
	const permutation<N> &perm) {

	block_index_space<N> bis1(bis);
	bis1.permute(perm);
	return bis1;
}


template<size_t N>
void btod_copy<N>::do_perform_nozero(block_tensor_i<N, double> &btb,
	double c) throw(exception) {

	block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(btb);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();
	dimensions<N> bidimsb = m_bis.get_block_index_dims();

	//	Retain a copy of old symmetry in B
	//
	symmetry<N, double> symb_old(m_bis);
	so_copy<N, double>(ctrlb.req_const_symmetry()).perform(symb_old);

	//	Install a new symmetry in B
	//
	permutation<N> p0;
	so_add<N, double>(m_sym, p0, symb_old, p0).
		perform(ctrlb.req_symmetry());

	orbit_list<N, double> olb1(symb_old), olb2(ctrlb.req_const_symmetry());
	permutation<N> pinv(m_perm, true);

	//	First go over blocks in b which got "downgraded",
	//	that is turned unique from being replicas
	//
	for(typename orbit_list<N, double>::iterator iob = olb2.begin();
		iob != olb2.end(); iob++) {

		if(olb1.contains(olb2.get_abs_index(iob))) continue;

		//	Index of current block in A and B
		//
		index<N> ib(olb2.get_index(iob)), ia(ib);
		ia.permute(pinv);

		//	Canonical indexes of current block
		//
		orbit<N, double> oa(ctrla.req_const_symmetry(), ia);
		orbit<N, double> ob(symb_old, ib);
		abs_index<N> cia(oa.get_abs_canonical_index(), bidimsa),
			cib(ob.get_abs_canonical_index(), bidimsb);

		bool zeroa = ctrla.req_is_zero_block(cia.get_index());
		bool zerob = ctrlb.req_is_zero_block(cib.get_index());

		if(zeroa && zerob) {
			ctrlb.req_zero_block(ib);
			continue;
		}

		if(zeroa) {
			tensor_i<N, double> &blkb_can =
				ctrlb.req_block(cib.get_index());
			tensor_i<N, double> &blkb = ctrlb.req_block(ib);

			const transf<N, double> &trb = ob.get_transf(ib);

			tod_copy<N>(blkb_can, trb.get_perm(), trb.get_coeff()).
				perform(blkb);

			ctrlb.ret_block(ib);
			ctrlb.ret_block(cib.get_index());
			continue;
		}

		if(zerob) {
			tensor_i<N, double> &blka_can =
				ctrla.req_block(cia.get_index());
			tensor_i<N, double> &blkb = ctrlb.req_block(ib);

			const transf<N, double> &tra = oa.get_transf(ia);

			permutation<N> pa(tra.get_perm());
			pa.permute(m_perm);
			double ca = tra.get_coeff() * m_c;
			tod_copy<N>(blka_can, pa, ca * c).perform(blkb);

			ctrlb.ret_block(ib);
			ctrla.ret_block(cia.get_index());
			continue;
		}

		tensor_i<N, double> &blka_can =
			ctrla.req_block(cia.get_index());
		tensor_i<N, double> &blkb_can =
			ctrlb.req_block(cib.get_index());
		tensor_i<N, double> &blkb = ctrlb.req_block(ib);

		const transf<N, double> &tra = oa.get_transf(ia);
		const transf<N, double> &trb = ob.get_transf(ib);

		permutation<N> pa(tra.get_perm());
		pa.permute(m_perm);
		double ca = tra.get_coeff() * m_c;
		tod_add<N> oper(blkb_can, trb.get_perm(), trb.get_coeff());
		oper.add_op(blka_can, pa, ca * c);
		oper.perform(blkb);

		ctrlb.ret_block(ib);
		ctrlb.ret_block(cib.get_index());
		ctrla.ret_block(cia.get_index());
	}

	//	Go over blocks in B that stay canonical
	//
	for(typename orbit_list<N, double>::iterator iob = olb2.begin();
		iob != olb2.end(); iob++) {

		if(!olb1.contains(olb2.get_abs_index(iob))) continue;

		//	Index of current block in A and B
		//
		index<N> ib(olb2.get_index(iob)), ia(ib);
		ia.permute(pinv);

		//	Canonical indexes of current block
		//
		orbit<N, double> oa(ctrla.req_const_symmetry(), ia);
		abs_index<N> cia(oa.get_abs_canonical_index(), bidimsa);

		bool zeroa = ctrla.req_is_zero_block(cia.get_index());
		bool zerob = ctrlb.req_is_zero_block(ib);

		if(zeroa) {
			continue;
		}

		tensor_i<N, double> &blka_can =
			ctrla.req_block(cia.get_index());
		tensor_i<N, double> &blkb = ctrlb.req_block(ib);

		const transf<N, double> &tra = oa.get_transf(ia);
		permutation<N> pa(tra.get_perm());
		pa.permute(m_perm);

		double ca = tra.get_coeff() * m_c;
		if(zerob) {
			tod_copy<N>(blka_can, pa, ca * c).perform(blkb);
		} else {
			tod_copy<N>(blka_can, pa, ca).perform(blkb, c);
		}

		ctrlb.ret_block(ib);
		ctrla.ret_block(cia.get_index());
	}

}


template<size_t N>
void btod_copy<N>::do_perform_zero(block_tensor_i<N, double> &btb,
	double c) throw(exception) {

	block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(btb);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	so_copy<N, double>(m_sym).perform(ctrlb.req_symmetry());

	orbit_list<N, double> ola(ctrla.req_const_symmetry());
	for(typename orbit_list<N, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

		//	Canonical index in A and corresponding index in B
		index<N> bia(ola.get_index(ioa)), bib1(bia);
		bib1.permute(m_perm);

		//	Find the canonical index in B
		orbit<N, double> ob(ctrlb.req_symmetry(), bib1);
		abs_index<N> abib(ob.get_abs_canonical_index(), bidims);
		index<N> bib(abib.get_index());

		//	Reverse transformation for block in B
		const transf<N, double> &trb = ob.get_transf(bib1);
		permutation<N> permb(m_perm);
		permb.permute(permutation<N>(trb.get_perm(), true));
		double cb = m_c / trb.get_coeff();

		//	Block A[bia] transformed with (permb, cb) will be
		//	copied to B[bib]

		bool zeroa = ctrla.req_is_zero_block(bia);
		bool zerob = ctrlb.req_is_zero_block(bib);

		if(zeroa) {
			ctrlb.req_zero_block(bib);
			continue;
		}

		tensor_i<N, double> &blka = ctrla.req_block(bia);
		tensor_i<N, double> &blkb = ctrlb.req_block(bib);

		tod_copy<N>(blka, permb, cb * c).perform(blkb);

		ctrla.ret_block(bia);
		ctrlb.ret_block(bib);
	}
}


template<size_t N>
void btod_copy<N>::do_perform(block_tensor_i<N, double> &bt,
	const index<N> &dst_blk_idx, bool zero, double c) throw(exception) {

	block_tensor_ctrl<N, double> src_ctrl(m_bta), dst_ctrl(bt);

	permutation<N> invperm(m_perm, true);
	index<N> src_blk_idx(dst_blk_idx);
	src_blk_idx.permute(invperm);

	if(src_ctrl.req_is_zero_block(src_blk_idx)) {

		if(zero) dst_ctrl.req_zero_block(dst_blk_idx);

	} else {

		bool adjzero = zero || dst_ctrl.req_is_zero_block(dst_blk_idx);
		tensor_i<N, double> &src_blk = src_ctrl.req_block(src_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);

		if(adjzero) {
			tod_copy<N> cp(src_blk, m_perm, m_c * c);
			cp.perform(dst_blk);
		} else {
			tod_copy<N> cp(src_blk, m_perm, m_c);
			cp.perform(dst_blk, c);
		}
	}

	src_ctrl.ret_block(src_blk_idx);
	dst_ctrl.ret_block(dst_blk_idx);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_H