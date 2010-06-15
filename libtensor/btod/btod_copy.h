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

	using additive_btod<N>::perform;

	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual const assignment_schedule<N, double> &get_schedule() const {
		return m_sch;
	}
	//@}

protected:
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &ib);
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &ib, const transf<N, double> &tr, double c);

private:
	static block_index_space<N> mk_bis(const block_index_space<N> &bis,
		const permutation<N> &perm);
	void make_schedule();

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

	if(!ctrla.req_is_zero_block(acia.get_index())) {
		tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
		tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(blk);
		ctrla.ret_block(acia.get_index());
	} else {
		tod_set<N>().perform(blk);
	}
}


template<size_t N>
void btod_copy<N>::compute_block(tensor_i<N, double> &blk, const index<N> &ib,
	const transf<N, double> &tr, double c) {

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
	tra.transform(tr);

	if(!ctrla.req_is_zero_block(acia.get_index())) {
		tensor_i<N, double> &blka = ctrla.req_block(acia.get_index());
		tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(blk, c);
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

#endif // LIBTENSOR_BTOD_COPY_H
