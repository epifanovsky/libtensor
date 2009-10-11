#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <map>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "core/orbit_list.h"
#include "tod/tod_copy.h"
#include "btod_additive.h"
#include "btod_so_copy.h"
#include "btod_so_equalize.h"

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
	typedef timings< btod_copy<N> > timings_base;

private:
	block_tensor_i<N, double> &m_bt; //!< Source block %tensor
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
	btod_copy(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Initializes the permuted copy operation
		\param bt Source block %tensor.
		\param p Permutation.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_copy();
	//@}

	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	virtual const symmetry<N, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<N, double> &bt) throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double c)
		throw(exception);
	//@}

private:
	void do_perform(block_tensor_i<N, double> &bt, bool zero, double c)
		throw(exception);
	void do_perform(block_tensor_i<N, double> &bt, const index<N> &dst_idx,
		bool zero, double c) throw(exception);

private:
	btod_copy<N> &operator=(const btod_copy<N>&);

};


template<size_t N>
const char *btod_copy<N>::k_clazz = "btod_copy<N>";


template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bt, double c)
: m_bt(bt), m_c(c), m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims()),
	m_sym(m_bis) {

	block_tensor_ctrl<N, double> ctrl(bt);
	m_sym.set_union(ctrl.req_symmetry());
}


template<size_t N>
btod_copy<N>::btod_copy(
	block_tensor_i<N, double> &bt, const permutation<N> &p, double c)
: m_bt(bt), m_perm(p), m_c(c), m_bis(bt.get_bis()),
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis) {

	block_tensor_ctrl<N, double> ctrl(bt);
	m_sym.set_union(ctrl.req_symmetry());

	m_bis.permute(m_perm);
	m_bidims.permute(m_perm);
	m_sym.permute(m_perm);
}


template<size_t N>
btod_copy<N>::~btod_copy() {

}


template<size_t N>
inline const block_index_space<N> &btod_copy<N>::get_bis() const {

	return m_bis;
}


template<size_t N>
inline const symmetry<N, double> &btod_copy<N>::get_symmetry() const {

	return m_sym;
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	timings_base::start_timer();

	btod_so_copy<N> symcopy(m_sym);
	symcopy.perform(bt);

	do_perform(bt, true, 1.0);

	timings_base::stop_timer();
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, const index<N>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}
	if(!m_bidims.contains(idx)) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Block index is out of bounds.");
	}

	timings_base::start_timer();
	do_perform(bt, idx, true, 1.0);
	timings_base::stop_timer();
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(fabs(c) == 0.0) return;

	timings_base::start_timer();

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block index space of the output tensor.");
	}

	block_tensor_ctrl<N, double> dst_ctrl(bt);
	const symmetry<N, double> &dst_sym = dst_ctrl.req_symmetry();

	symmetry<N, double> sym(m_sym);

	if(sym.equals(dst_sym)) {
		// Sym(A) = Sym(B)
		do_perform(bt, false, c);
	} else {
		sym.set_intersection(dst_sym);
		if(sym.equals(m_sym)) {
			// Sym(A) < Sym(B)
			throw_exc(k_clazz, method,
				"Case S(A)<S(B) is not handled.");
		} else if(sym.equals(dst_sym)) {
			// Sym(B) < Sym(A)
			throw_exc(k_clazz, method,
				"Case S(B)<S(A) is not handled.");
		} else {
			// Sym(A) > Sym'(A) = Sym'(B) < Sym(B)
			throw_exc(k_clazz, method,
				"Case S(A)>S'(A)=S'(B)<S(B) is not handled.");
		}
	}

	timings_base::stop_timer();
}


template<size_t N>
void btod_copy<N>::do_perform(
	block_tensor_i<N, double> &bt, bool zero, double c) throw(exception) {

	block_tensor_ctrl<N, double> src_ctrl(m_bt), dst_ctrl(bt);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	orbit_list<N, double> orblst(src_ctrl.req_symmetry());
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {

		index<N> src_blk_idx(orblst.get_index(iorbit));
		if(src_ctrl.req_is_zero_block(src_blk_idx)) continue;
		index<N> dst_blk_idx(src_blk_idx);
		dst_blk_idx.permute(m_perm);
		orbit<N, double> dst_orb(dst_ctrl.req_symmetry(), dst_blk_idx);
		const transf<N, double> &dst_trn =
			dst_orb.get_transf(dst_blk_idx);
		permutation<N> perm(m_perm);
		perm.permute(permutation<N>(dst_trn.get_perm(), true));
		double coeff = m_c / dst_trn.get_coeff();

		tensor_i<N, double> &src_blk = src_ctrl.req_block(src_blk_idx);
		index<N> dst_blk_can_idx;
		bidims.abs_index(
			dst_orb.get_abs_canonical_index(), dst_blk_can_idx);
		tensor_i<N, double> &dst_blk =
			dst_ctrl.req_block(dst_blk_can_idx);

		if(zero) {
			tod_copy<N> cp(src_blk, m_perm, coeff * c);
			cp.perform(dst_blk);
		} else {
			tod_copy<N> cp(src_blk, m_perm, coeff);
			cp.perform(dst_blk, c);
		}

		src_ctrl.ret_block(src_blk_idx);
		dst_ctrl.ret_block(dst_blk_idx);

	}
}


template<size_t N>
void btod_copy<N>::do_perform(block_tensor_i<N, double> &bt,
	const index<N> &dst_blk_idx, bool zero, double c) throw(exception) {

	block_tensor_ctrl<N, double> src_ctrl(m_bt), dst_ctrl(bt);

	permutation<N> invperm(m_perm, true);
	index<N> src_blk_idx(dst_blk_idx);
	src_blk_idx.permute(invperm);

	if(src_ctrl.req_is_zero_block(src_blk_idx)) {

		if(zero) dst_ctrl.req_zero_block(dst_blk_idx);

	} else {

		tensor_i<N, double> &src_blk = src_ctrl.req_block(src_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);

		if(zero) {
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

#endif // LITENSOR_BTOD_COPY_H
