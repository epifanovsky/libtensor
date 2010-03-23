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
#include "../symmetry/so_copy.h"
#include "../symmetry/so_permute.h"
#include "../tod/tod_copy.h"
#include "bad_block_index_space.h"
#include "btod_additive.h"
//~ #include "btod_so_copy.h"
#include "../not_implemented.h"

namespace libtensor {


/**	\brief Makes a copy of a block %tensor, applying a permutation and
		a scaling coefficient
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public btod_additive<N>, public timings< btod_copy<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bta; //!< Source block %tensor
	permutation<N> m_perm; //!< Permutation
	double m_c; //!< Scaling coefficient
	block_index_space<N> m_bis; //!< Block %index space of output
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_sym; //!< Symmetry of output

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
	virtual ~btod_copy();
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

	virtual void perform(block_tensor_i<N, double> &btb) throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &btb, double c)
		throw(exception);
	//@}

private:
	static block_index_space<N> mk_bis(const block_index_space<N> &bis,
		const permutation<N> &perm);
	void do_perform(block_tensor_i<N, double> &btb, double c)
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
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	so_copy<N, double>(ctrla.req_const_symmetry()).perform(m_sym);
}


template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
	double c) :

	m_bta(bta), m_perm(p), m_c(c), m_bis(mk_bis(m_bta.get_bis(), m_perm)),
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis) {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	symmetry<N, double> sym1(m_bta.get_bis());
	so_copy<N, double>(ctrla.req_const_symmetry()).perform(sym1);
	so_permute<N, double>(sym1, m_perm).perform(m_sym);
}


template<size_t N>
btod_copy<N>::~btod_copy() {

}


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
void btod_copy<N>::perform(block_tensor_i<N, double> &btb, double c)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!m_bis.equals(btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}

	if(fabs(c) == 0.0) return;

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);

	btod_copy<N>::start_timer();

	//~ block_tensor_ctrl<N, double> dst_ctrl(bt);
	//~ const symmetry<N, double> &dst_sym = dst_ctrl.req_symmetry();

	//~ symmetry<N, double> sym(m_sym);

	//~ if(sym.equals(dst_sym)) {
		//~ // Sym(A) = Sym(B)
		//~ do_perform(bt, false, c);
	//~ } else {
		//~ sym.set_intersection(dst_sym);
		//~ if(sym.equals(m_sym)) {
			//~ // Sym(A) < Sym(B)
			//~ throw_exc(k_clazz, method,
				//~ "Case S(A)<S(B) is not handled.");
		//~ } else if(sym.equals(dst_sym)) {
			//~ // Sym(B) < Sym(A)
			//~ throw_exc(k_clazz, method,
				//~ "Case S(B)<S(A) is not handled.");
		//~ } else {
			//~ // Sym(A) > Sym'(A) = Sym'(B) < Sym(B)
			//~ throw_exc(k_clazz, method,
				//~ "Case S(A)>S'(A)=S'(B)<S(B) is not handled.");
		//~ }
	//~ }

	btod_copy<N>::stop_timer();
}


template<size_t N>
block_index_space<N> btod_copy<N>::mk_bis(const block_index_space<N> &bis,
	const permutation<N> &perm) {

	block_index_space<N> bis1(bis);
	bis1.permute(perm);
	return bis1;
}


//~ template<size_t N>
//~ void btod_copy<N>::do_perform(block_tensor_i<N, double> &btb,
	//~ double c) throw(exception) {

	//~ block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(btb);
	//~ dimensions<N> bidims = m_bis.get_block_index_dims();

	//~ so_intersection<N>();

	//~ orbit_list<N, double> ola(ctrla.req_const_symmetry());
	//~ for(typename orbit_list<N, double>::iterator ioa = ola.begin();
		//~ ioa != ola.end(); ioa++) {

		//~ index<N> bia(ola.get_index(ioa)), bib(bia);
		//~ bib.permute(m_perm);

		//~ bool zeroa = ctrla.req_is_zero_block(bia);
		//~ if(zeroa) {
			//~ continue;
		//~ }

		//~ orbit<N, double> ob(ctrlb.req_symmetry(), bib);
		//~ const transf<N, double> &dst_trn = ob.get_transf(bib);
		//~ permutation<N> perm(m_perm);
		//~ perm.permute(permutation<N>(dst_trn.get_perm(), true));
		//~ double coeff = m_c / dst_trn.get_coeff();

		//~ tensor_i<N, double> &blka = ctrla.req_block(bia);
		//~ index<N> dst_blk_can_idx;
		//~ bidims.abs_index(ob.get_abs_canonical_index(), dst_blk_can_idx);

		//~ bool adjzero = ctrlb.req_is_zero_block(dst_blk_can_idx);
		//~ tensor_i<N, double> &blkb =
			//~ ctrlb.req_block(dst_blk_can_idx);

		//~ if(adjzero) {
			//~ tod_copy<N>(blka, m_perm, coeff * c).perform(blkb);
		//~ } else {
			//~ tod_copy<N> cp(blka, m_perm, coeff).perform(blkb, c);
		//~ }

		//~ ctrla.ret_block(bia);
		//~ ctrlb.ret_block(bib);
	//~ }
//~ }


template<size_t N>
void btod_copy<N>::do_perform_zero(block_tensor_i<N, double> &btb,
	double c) throw(exception) {

	block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(btb);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	so_copy<N, double>(ctrla.req_const_symmetry()).perform(
		ctrlb.req_symmetry());

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

		if(zerob) {
			tod_copy<N>(blka, permb, cb * c).perform(blkb);
		} else {
			tod_copy<N>(blka, permb, cb).perform(blkb, c);
		}

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