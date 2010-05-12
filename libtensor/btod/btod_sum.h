#ifndef LIBTENSOR_BTOD_SUM_H
#define LIBTENSOR_BTOD_SUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "additive_btod.h"
#include "../not_implemented.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Adds results of a %sequence of block %tensor operations
		(for double)
	\tparam N Tensor order.

	This operation runs a %sequence of block %tensor operations and
	accumulates their results with given coefficients. All of the operations
	in the %sequence shall derive from additive_btod<N>.

	The %sequence must contain at least one operation, which is called the
	base operation.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_sum :
	public additive_btod<N>,
	public timings< btod_sum<N> > {

public:
	static const char* k_clazz; //!< Class name

private:
	//!	\brief List node type
	typedef struct node {
	private:
		additive_btod<N> *m_op;
		double m_c;
	public:
		node() : m_op(NULL), m_c(0.0) { }
		node(additive_btod<N> &op, double c) : m_op(&op), m_c(c) { }
		additive_btod<N> &get_op() { return *m_op; }
		double get_coeff() const { return m_c; }
	} node_t;

private:
	mutable std::list<node_t> m_ops; //!< List of operations
	block_index_space<N> m_bis; //!< Block index space
	dimensions<N> m_bidims; //!< Block index dims
	symmetry<N, double> m_sym; //!< Symmetry of operation
	mutable bool m_dirty_sch; //!< Whether the assignment schedule is dirty
	mutable assignment_schedule<N, double> *m_sch; //!< Assignment schedule

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the base operation
		\param op Operation.
		\param c Coefficient.
	 **/
	btod_sum(additive_btod<N> &op, double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_sum();

	//@}


	//!	\name Implementation of libtensor::direct_tensor_operation<N>
	//@{

	virtual const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N, double> &get_schedule() const {
		if(m_sch == 0 || m_dirty_sch) make_schedule();
		return *m_sch;
	}

	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}


	//!	\name Implementation of libtensor::additive_btod<N>
	//@{
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &i);
	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		const transf<N, double> &tr, double c);

	using additive_btod<N>::perform;

	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx,
		double c) throw(exception);
	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Adds an operation to the sequence
		\param op Operation.
		\param c Coefficient.
	 **/
	void add_op(additive_btod<N> &op, double c = 1.0);

	//@}

private:
	void make_schedule() const;

private:
	btod_sum<N> &operator=(const btod_sum<N>&);

};


template<size_t N>
const char* btod_sum<N>::k_clazz = "btod_sum<N>";


template<size_t N>
inline btod_sum<N>::btod_sum(additive_btod<N> &op, double c) :
	m_bis(op.get_bis()), m_bidims(m_bis.get_block_index_dims()),
	m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

	add_op(op, c);
}


template<size_t N>
btod_sum<N>::~btod_sum() {

	delete m_sch;
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&, "
			"const index<N>)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}
	if(!m_bis.get_block_index_dims().contains(idx)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__,
			__LINE__, "Invalid block index.");
	}

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx,
	double c) throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, const index<N>&, double)";

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N>
void btod_sum<N>::compute_block(tensor_i<N, double> &blk, const index<N> &i) {

	abs_index<N> ai(i, m_bidims);
	transf<N, double> tr0;

	tod_set<N>().perform(blk);

	for(typename std::list<node_t>::iterator iop = m_ops.begin();
		iop != m_ops.end(); iop++) {

		if(iop->get_op().get_schedule().contains(ai.get_abs_index())) {
			additive_btod<N>::compute_block(iop->get_op(), blk, i,
				tr0, iop->get_coeff());
		}
	}
}


template<size_t N>
void btod_sum<N>::compute_block(tensor_i<N, double> &blk, const index<N> &i,
	const transf<N, double> &tr, double c) {

	abs_index<N> ai(i, m_bidims);

	for(typename std::list<node_t>::iterator iop = m_ops.begin();
		iop != m_ops.end(); iop++) {

		if(iop->get_op().get_schedule().contains(ai.get_abs_index())) {
			additive_btod<N>::compute_block(iop->get_op(), blk, i,
				tr, c * iop->get_coeff());
		}
	}
}


template<size_t N>
void btod_sum<N>::add_op(additive_btod<N> &op, double c) {

	static const char *method = "add_op(additive_btod<N>&, double)";

	if(!op.get_bis().equals(m_bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "op");
	}

	m_ops.push_back(node_t(op, c));
}


template<size_t N>
void btod_sum<N>::make_schedule() const {

	delete m_sch;
	m_sch = new assignment_schedule<N, double>(m_bidims);

	for(typename std::list<node_t>::iterator iop = m_ops.begin();
		iop != m_ops.end(); iop++) {

		const assignment_schedule<N, double> &sch =
			iop->get_op().get_schedule();
		for(typename assignment_schedule<N, double>::iterator j =
			sch.begin(); j != sch.end(); j++) {

			if(!m_sch->contains(sch.get_abs_index(j))) {
				m_sch->insert(sch.get_abs_index(j));
			}
		}
	}

	m_dirty_sch = false;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_H

