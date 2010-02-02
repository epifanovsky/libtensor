#ifndef LIBTENSOR_BTOD_ADD_H
#define LIBTENSOR_BTOD_ADD_H

#include <cmath>
#include <list>
#include <new>
#include <vector>
#include <utility>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include "bad_block_index_space.h"
#include "btod_additive.h"
#include "btod_so_copy.h"

namespace libtensor {


/**	\brief Addition of multiple block tensors

	This block %tensor operation performs the addition of block tensors:
	\f[ B = c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots \f]

	The operation must have at least one operand provided at the time of
	construction. Other operands are added afterwards and must agree in
	the dimensions and the block structure.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_add :
	public btod_additive<N>,
	public timings< btod_add<N> > {

public:
	static const char *k_clazz; //!< Class name

private:
	typedef timings< btod_add<N> > timings_base;

private:
	typedef struct operand {
		block_tensor_i<N, double> &m_bt; //!< Block %tensor
		permutation<N> m_perm; //!< Permutation
		double m_c; //!< Scaling coefficient
		operand(block_tensor_i<N, double> &bt,
			const permutation<N> &perm, double c)
		: m_bt(bt), m_perm(perm), m_c(c) { };
	} operand_t;

	typedef struct {
		block_tensor_ctrl<N, double> *m_ctrl;
		index<N> m_idx;
		transf<N, double> m_tr;
	} arg_t;

	block_index_space<N> m_bis; //!< Block %index space of the result
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_sym; //!< Symmetry of the result
	std::vector<operand_t*> m_ops; //!< Operand list

public:
	//!	\name Construction, destruction, initialization
	//@{

	/**	\brief Initializes the addition operation
		\param bt First block %tensor in the series.
		\param c Scaling coefficient.
	 **/
	btod_add(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Initializes the addition operation
		\param bt First block %tensor in the series.
		\param pb Permutation of the first %tensor.
		\param c Scaling coefficient.
	 **/
	btod_add(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_add();

	/**	\brief Adds an operand (block %tensor in the series)
		\param bt Block %tensor.
		\param c Scaling coefficient.
		\throw bad_parameter If the block %tensor has incompatible
			%dimensions or block structure.
		\throw out_of_memory If memory allocation fails.
	 **/
	void add_op(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Adds an operand (block %tensor in the series)
		\param bt Block %tensor.
		\param perm Permutation of the block %tensor.
		\param c Scaling coefficient.
		\throw bad_parameter If the block %tensor has incompatible
			%dimensions or block structure.
		\throw out_of_memory If memory allocation fails.
	 **/
	void add_op(block_tensor_i<N, double> &bt, const permutation<N> &perm,
		double c = 1.0);

	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	virtual const symmetry<N, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<N, double> &bt)
		throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double cb)
		throw(exception);
	//@}

private:
	void add_operand(block_tensor_i<N, double> &bt,
		const permutation<N> &perm, double c, bool adjsym)
		throw(out_of_memory);

	void do_perform(block_tensor_i<N, double> &bt, bool zero, double cb);

	void process_list(block_tensor_ctrl<N, double> &dst_ctrl,
		const index<N> &dst_blk_idx, const std::list<arg_t> &lst,
		bool zero, double c);

private:
	btod_add<N> &operator=(const btod_add<N>&);

};


template<size_t N>
const char *btod_add<N>::k_clazz = "btod_add<N>";


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, double c) :
	m_bis(bt.get_bis()),
	m_bidims(m_bis.get_block_index_dims()),
	m_sym(m_bis) {

	m_bis.match_splits();
	add_operand(bt, permutation<N>(), c, false);
	block_tensor_ctrl<N, double> ctrl(bt);
	m_sym.set_union(ctrl.req_symmetry());
}


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt,
	const permutation<N> &perm, double c) :
		m_bis(bt.get_bis()),
		m_bidims(m_bis.get_block_index_dims().permute(perm)),
		m_sym(m_bis) {

	m_bis.match_splits();
	add_operand(bt, perm, c, false);
	block_tensor_ctrl<N, double> ctrl(bt);
	m_sym.set_union(ctrl.req_symmetry());

	m_bis.permute(perm);
	m_bidims.permute(perm);
	m_sym.permute(perm);
}


template<size_t N>
btod_add<N>::~btod_add() {

	typename std::vector<operand_t*>::iterator i = m_ops.begin();
	while(i != m_ops.end()) {
		delete (*i);
		*i = NULL;
		i++;
	}
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt, double c) {

	static const char *method =
		"add_op(block_tensor_i<N, double>&, double)";

	if(fabs(c) == 0.0) return;

	block_index_space<N> bis(bt.get_bis());
	bis.match_splits();
	if(!m_bis.equals(bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "bt");
	}

	add_operand(bt, permutation<N>(), c, true);
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt,
	const permutation<N> &perm, double c) {

	static const char *method = "add_op(block_tensor_i<N, double>&, "
		"const permutation<N>&, double)";

	if(fabs(c) == 0.0) return;

	block_index_space<N> bis(bt.get_bis());
	bis.match_splits();
	bis.permute(perm);
	if(!m_bis.equals(bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "bt");
	}

	add_operand(bt, perm, c, true);
}


template<size_t N>
const block_index_space<N> &btod_add<N>::get_bis() const {

	return m_bis;
}


template<size_t N>
const symmetry<N, double> &btod_add<N>::get_symmetry() const {

	return m_sym;
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	block_index_space<N> bis(bt.get_bis());
	bis.match_splits();
	if(!m_bis.equals(bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "bt");
	}

	timings_base::start_timer();

	btod_so_copy<N> symcopy(m_sym);
	symcopy.perform(bt);

	do_perform(bt, true, 1.0);

	timings_base::stop_timer();
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	throw_exc(k_clazz, "perform(const index<N>&)", "NIY");
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt, double cb)
	throw(exception){

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	block_index_space<N> bis(bt.get_bis());
	bis.match_splits();
	if(!m_bis.equals(bis)) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	timings_base::start_timer();

	block_tensor_ctrl<N, double> dst_ctrl(bt);
	const symmetry<N, double> &dst_sym = dst_ctrl.req_symmetry();

	symmetry<N, double> sym(m_sym);

	if(sym.equals(dst_sym)) {
		// Sym(A) = Sym(B)
		do_perform(bt, false, cb);
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
void btod_add<N>::add_operand(block_tensor_i<N, double> &bt,
	const permutation<N> &perm, double c, bool adjsym)
	throw(out_of_memory) {

	static const char *method = "add_operand(block_tensor_i<N,double>&, "
		"const permutation<N>&, double)";

	try {
		operand_t *op = new operand(bt, perm, c);
		m_ops.push_back(op);
	} catch(std::bad_alloc &e) {
		throw out_of_memory("libtensor", k_clazz, method, __FILE__,
			__LINE__, "op");
	}

	if(adjsym) {
		block_tensor_ctrl<N, double> ctrl(bt);
		symmetry<N, double> sym(ctrl.req_symmetry());
		sym.permute(perm);
		m_sym.set_intersection(sym);
	}
}


template<size_t N>
void btod_add<N>::do_perform(
	block_tensor_i<N, double> &bt, bool zero, double cb) {

	block_tensor_ctrl<N, double> dst_ctrl(bt);
	std::vector< block_tensor_ctrl<N, double>* > src_ctrl(
		m_ops.size(), NULL);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	for(size_t iop = 0; iop < m_ops.size(); iop++) {
		src_ctrl[iop] =
			new block_tensor_ctrl<N, double>(m_ops[iop]->m_bt);
	}

	orbit_list<N, double> orblst(m_sym);
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {

		const index<N> &dst_blk_idx = orblst.get_index(iorbit);
		std::list<arg_t> arglst;

		for(size_t iop = 0; iop < m_ops.size(); iop++) {
			block_tensor_ctrl<N, double> &ctrl = *(src_ctrl[iop]);
			operand_t &op = *(m_ops[iop]);
			permutation<N> perm(op.m_perm),
				invperm(op.m_perm, true);

			index<N> src_blk_idx(dst_blk_idx), can_blk_idx;
			src_blk_idx.permute(invperm);

			orbit<N, double> orb(ctrl.req_symmetry(), src_blk_idx);
			dimensions<N> bidims(ctrl.req_symmetry().get_bis().
				get_block_index_dims());
			bidims.abs_index(
				orb.get_abs_canonical_index(), can_blk_idx);
			transf<N, double> tr(orb.get_transf(src_blk_idx));
			if(ctrl.req_is_zero_block(can_blk_idx)) continue;
			tensor_i<N, double> &src_blk =
				ctrl.req_block(can_blk_idx);
			tr.get_perm().permute(perm);
			tr.get_coeff() *= op.m_c;

			if(tr.get_coeff() != 0.0) {
				arg_t arg;
				arg.m_ctrl = &ctrl;
				arg.m_idx = can_blk_idx;
				arg.m_tr = tr;
				arglst.push_back(arg);
			}
		}
		process_list(dst_ctrl, dst_blk_idx, arglst, zero, cb);

	}

	for(size_t iop = 0; iop < m_ops.size(); iop++) {
		delete src_ctrl[iop];
		src_ctrl[iop] = NULL;
	}

}


template<size_t N>
void btod_add<N>::process_list(block_tensor_ctrl<N, double> &dst_ctrl,
	const index<N> &dst_blk_idx, const std::list<arg_t> &lst,
	bool zero, double c) {

	size_t lstsz = lst.size();
	if(lstsz == 1) {

		typename std::list<arg_t>::const_iterator iarg = lst.begin();
		tensor_i<N, double> &src_blk =
			iarg->m_ctrl->req_block(iarg->m_idx);
		bool adjzero = zero || dst_ctrl.req_is_zero_block(dst_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);

		tod_copy<N> todcp(src_blk, iarg->m_tr.get_perm(),
			iarg->m_tr.get_coeff() * c);
		if(adjzero) todcp.perform(dst_blk);
		else todcp.perform(dst_blk, 1.0);

		iarg->m_ctrl->ret_block(iarg->m_idx);
		dst_ctrl.ret_block(dst_blk_idx);

	} else if(lstsz > 1) {

		typename std::list<arg_t>::const_iterator iarg = lst.begin();

		tod_add<N> todadd(iarg->m_ctrl->req_block(iarg->m_idx),
			iarg->m_tr.get_perm(), iarg->m_tr.get_coeff() * c);

		for(iarg++; iarg != lst.end(); iarg++) {
			tensor_i<N, double> &src_blk =
				iarg->m_ctrl->req_block(iarg->m_idx);
			todadd.add_op(src_blk, iarg->m_tr.get_perm(),
				iarg->m_tr.get_coeff() * c);
		}

		bool adjzero = zero || dst_ctrl.req_is_zero_block(dst_blk_idx);
		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);
		if(adjzero) todadd.perform(dst_blk);
		else todadd.perform(dst_blk, 1.0);

		for(iarg = lst.begin(); iarg != lst.end(); iarg++) {
			iarg->m_ctrl->ret_block(iarg->m_idx);
		}
		dst_ctrl.ret_block(dst_blk_idx);

	} else {

		if(zero) dst_ctrl.req_zero_block(dst_blk_idx);

	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_H
