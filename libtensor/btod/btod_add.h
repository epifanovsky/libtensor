#ifndef LIBTENSOR_BTOD_ADD_H
#define LIBTENSOR_BTOD_ADD_H

#include <cmath>
#include <list>
#include <new>
#include <vector>
#include <utility>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_add.h"
#include "../symmetry/so_copy.h"
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
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
	public additive_btod<N>,
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

	struct schrec {
		size_t iarg;
		index<N> idx;
		permutation<N> perm;
		double k;
	};

	block_index_space<N> m_bis; //!< Block %index space of the result
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_sym; //!< Symmetry of the result
	std::vector<operand_t*> m_ops; //!< Operand list
	mutable bool m_dirty_sch; //!< Whether the schedule is dirty
	mutable assignment_schedule<N, double> *m_sch; //!< Assignment schedule
	mutable std::multimap<size_t, schrec> m_op_sch; //!< Operation schedule

	typedef typename std::multimap<size_t, schrec>::const_iterator
		schiterator_t;

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

	virtual const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

	virtual assignment_schedule<N, double> &get_schedule() const {
		if(m_dirty_sch || m_sch == 0) make_schedule();
		return *m_sch;
	}

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i);

	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		const transf<N, double> &tr, double c);

	using additive_btod<N>::perform;

//	virtual void perform(block_tensor_i<N, double> &bt)
//		throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::additive_btod<N>
	//@{
//	virtual void perform(block_tensor_i<N, double> &bt, double cb)
//		throw(exception);
	//@}

private:
	void compute_block(tensor_i<N, double> &blkb,
		const std::pair<schiterator_t, schiterator_t> ipair, bool zero,
		const transf<N, double> &trb, double kb);

	void add_operand(block_tensor_i<N, double> &bt,
		const permutation<N> &perm, double c);

	void make_schedule() const;

//	void do_perform(block_tensor_i<N, double> &bt, bool zero, double cb);

//	void process_list(block_tensor_ctrl<N, double> &dst_ctrl,
//		const index<N> &dst_blk_idx, const std::list<arg_t> &lst,
//		bool zero, double c);

private:
	btod_add(const btod_add<N>&);
	btod_add<N> &operator=(const btod_add<N>&);

};


template<size_t N>
const char *btod_add<N>::k_clazz = "btod_add<N>";


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, double c) :

	m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims()),
	m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

	m_bis.match_splits();
	add_operand(bt, permutation<N>(), c);
}


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, const permutation<N> &perm,
	double c) :

	m_bis(block_index_space<N>(bt.get_bis()).permute(perm)),
	m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_dirty_sch(true),
	m_sch(0) {

	m_bis.match_splits();
	add_operand(bt, perm, c);
}


template<size_t N>
btod_add<N>::~btod_add() {

	delete m_sch;

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

	add_operand(bt, permutation<N>(), c);
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

	add_operand(bt, perm, c);
}


//template<size_t N>
//void btod_add<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {
//
//	static const char *method = "perform(block_tensor_i<N, double>&)";
//
//	block_index_space<N> bis(bt.get_bis());
//	bis.match_splits();
//	if(!m_bis.equals(bis)) {
//		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
//			__LINE__, "bt");
//	}
//
//	timings_base::start_timer();
//
//	btod_so_copy<N> symcopy(m_sym);
//	symcopy.perform(bt);
//
//	do_perform(bt, true, 1.0);
//
//	timings_base::stop_timer();
//}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	throw_exc(k_clazz, "perform(const index<N>&)", "NIY");
}


//template<size_t N>
//void btod_add<N>::perform(block_tensor_i<N, double> &bt, double cb)
//	throw(exception){
//
//	static const char *method =
//		"perform(block_tensor_i<N, double>&, double)";
//
//	block_index_space<N> bis(bt.get_bis());
//	bis.match_splits();
//	if(!m_bis.equals(bis)) {
//		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
//			__LINE__, "Incompatible block index space.");
//	}
//
//	timings_base::start_timer();
//
//	block_tensor_ctrl<N, double> dst_ctrl(bt);
//	const symmetry<N, double> &dst_sym = dst_ctrl.req_symmetry();
//
//	symmetry<N, double> sym(m_sym);
//
//	if(sym.equals(dst_sym)) {
//		// Sym(A) = Sym(B)
//		do_perform(bt, false, cb);
//	} else {
//		sym.set_intersection(dst_sym);
//		if(sym.equals(m_sym)) {
//			// Sym(A) < Sym(B)
//			throw_exc(k_clazz, method,
//				"Case S(A)<S(B) is not handled.");
//		} else if(sym.equals(dst_sym)) {
//			// Sym(B) < Sym(A)
//			throw_exc(k_clazz, method,
//				"Case S(B)<S(A) is not handled.");
//		} else {
//			// Sym(A) > Sym'(A) = Sym'(B) < Sym(B)
//			throw_exc(k_clazz, method,
//				"Case S(A)>S'(A)=S'(B)<S(B) is not handled.");
//		}
//	}
//
//	timings_base::stop_timer();
//}


template<size_t N>
void btod_add<N>::compute_block(tensor_i<N, double> &blkb, const index<N> &ib) {

	static const char *method =
		"compute_block(tensor_i<N, double>&, const index<N>&)";

	btod_add<N>::start_timer();

	try {

		abs_index<N> aib(ib, m_bidims);
		std::pair<schiterator_t, schiterator_t> ipair =
			m_op_sch.equal_range(aib.get_abs_index());
		if(ipair.first == m_op_sch.end()) {
			tod_set<N>().perform(blkb);
		} else {
			transf<N, double> tr0;
			compute_block(blkb, ipair, true, tr0, 1.0);
		}

	} catch(...) {
		btod_add<N>::stop_timer();
		throw;
	}

	btod_add<N>::stop_timer();
}


template<size_t N>
void btod_add<N>::compute_block(tensor_i<N, double> &blkb, const index<N> &ib,
	const transf<N, double> &trb, double kb) {

	static const char *method = "tensor_i<N, double>&, const index<N>&, "
		"const transf<N, double>&, double)";

	btod_add<N>::start_timer();

	try {

		abs_index<N> aib(ib, m_bidims);
		std::pair<schiterator_t, schiterator_t> ipair =
			m_op_sch.equal_range(aib.get_abs_index());
		if(ipair.first != m_op_sch.end()) {
			compute_block(blkb, ipair, false, trb, kb);
		}

	} catch(...) {
		btod_add<N>::stop_timer();
		throw;
	}

	btod_add<N>::stop_timer();
}


template<size_t N>
void btod_add<N>::compute_block(tensor_i<N, double> &blkb,
	const std::pair<schiterator_t, schiterator_t> ipair, bool zero,
	const transf<N, double> &trb, double kb) {

	size_t narg = m_ops.size();
	std::vector<block_tensor_ctrl<N, double>*> ca(narg);
	for(size_t i = 0; i < narg; i++) {
		ca[i] = new block_tensor_ctrl<N, double>(m_ops[i]->m_bt);
	}

	schiterator_t iarg = ipair.first;
	tod_add<N> *op = 0;

	{
		const schrec &rec = iarg->second;
		permutation<N> perm(rec.perm); perm.permute(trb.get_perm());
		double k = rec.k * kb * trb.get_coeff();
		op = new tod_add<N>(ca[rec.iarg]->req_block(rec.idx), perm, k);
	}

	for(iarg++; iarg != ipair.second; iarg++) {
		const schrec &rec = iarg->second;
		permutation<N> perm(rec.perm); perm.permute(trb.get_perm());
		double k = rec.k * kb * trb.get_coeff();
		op->add_op(ca[rec.iarg]->req_block(rec.idx), perm, k);
	}

	if(zero) op->perform(blkb);
	else op->perform(blkb, 1.0);

	delete op;

	for(iarg = ipair.first; iarg != ipair.second; iarg++) {
		const schrec &rec = iarg->second;
		ca[rec.iarg]->ret_block(rec.idx);
	}

	for(size_t i = 0; i < narg; i++) delete ca[i];
}


template<size_t N>
void btod_add<N>::add_operand(block_tensor_i<N, double> &bt,
	const permutation<N> &perm, double c) {

	static const char *method = "add_operand(block_tensor_i<N,double>&, "
		"const permutation<N>&, double)";

	bool first = m_ops.empty();

	try {
		m_ops.push_back(new operand(bt, perm, c));
	} catch(std::bad_alloc &e) {
		throw out_of_memory("libtensor", k_clazz, method, __FILE__,
			__LINE__, "op");
	}

	block_tensor_ctrl<N, double> ca(bt);
	if(first) {
		so_copy<N, double>(ca.req_const_symmetry()).perform(m_sym);
	} else {
		symmetry<N, double> symcopy(m_bis);
		so_copy<N, double>(m_sym).perform(symcopy);
		so_add<N, double>(symcopy, permutation<N>(),
			ca.req_const_symmetry(), perm).perform(m_sym);
	}
	m_dirty_sch = true;
}


template<size_t N>
void btod_add<N>::make_schedule() const {

//	btod_add<N>::start_timer("make_schedule");

	delete m_sch;
	m_sch = new assignment_schedule<N, double>(m_bidims);
	m_op_sch.clear();

	size_t narg = m_ops.size();
	std::vector<block_tensor_ctrl<N, double>*> ca(narg);
	std::vector<orbit_list<N, double>*> ola(narg);

	for(size_t i = 0; i < narg; i++) {
		ca[i] = new block_tensor_ctrl<N, double>(m_ops[i]->m_bt);
		ola[i] = new orbit_list<N, double>(ca[i]->req_const_symmetry());
	}

	orbit_list<N, double> olb(m_sym);
	for(typename orbit_list<N, double>::iterator iob = olb.begin();
		iob != olb.end(); iob++) {

		size_t nrec = 0;

		for(size_t i = 0; i < narg; i++) {

			permutation<N> pinv(m_ops[i]->m_perm, true);
			index<N> ia(olb.get_index(iob)); ia.permute(pinv);
			dimensions<N> bidimsa(m_bidims); bidimsa.permute(pinv);
			abs_index<N> aia(ia, bidimsa);

			if(!ola[i]->contains(aia.get_abs_index())) {

				orbit<N, double> oa(ca[i]->req_const_symmetry(),
					ia);
				abs_index<N> acia(oa.get_abs_canonical_index(),
					bidimsa);

				if(ca[i]->req_is_zero_block(acia.get_index()))
					continue;

				const transf<N, double> &tra = oa.get_transf(
					aia.get_abs_index());

				schrec rec;
				rec.iarg = i;
				rec.idx = acia.get_index();
				rec.k = m_ops[i]->m_c * tra.get_coeff();
				rec.perm.permute(m_ops[i]->m_perm).
					permute(tra.get_perm());
				m_op_sch.insert(std::pair<size_t, schrec>(
					olb.get_abs_index(iob), rec));
				nrec++;
			} else {

				if(ca[i]->req_is_zero_block(ia)) continue;

				schrec rec;
				rec.iarg = i;
				rec.idx = aia.get_index();
				rec.k = m_ops[i]->m_c;
				rec.perm.permute(m_ops[i]->m_perm);
				m_op_sch.insert(std::pair<size_t, schrec>(
					olb.get_abs_index(iob), rec));
				nrec++;
			}
		}

		if(nrec > 0) m_sch->insert(olb.get_abs_index(iob));
	}

	for(size_t i = 0; i < narg; i++) {
		delete ola[i];
		delete ca[i];
	}

	m_dirty_sch = false;

//	btod_add<N>::stop_timer("make_schedule");
}

/*
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
		todcp.prefetch();
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
		todadd.prefetch();

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
*/

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_H
