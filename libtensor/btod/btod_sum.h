#ifndef LIBTENSOR_BTOD_SUM_H
#define LIBTENSOR_BTOD_SUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "btod_additive.h"
#include "../not_implemented.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Adds results of a %sequence of block %tensor operations
		(for double)
	\tparam N Tensor order.

	This operation runs a %sequence of block %tensor operations and
	accumulates their results with given coefficients. All of the operations
	in the %sequence shall derive from btod_additive<N>.

	The %sequence must contain at least one operation, which is called the
	base operation.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_sum :
	public btod_additive<N>,
	public timings< btod_sum<N> > {

public:
	static const char* k_clazz; //!< Class name

private:
	//!	\brief List node type
	typedef struct node {
	private:
		btod_additive<N> *m_op;
		double m_c;
	public:
		node() : m_op(NULL), m_c(0.0) { }
		node(btod_additive<N> &op, double c) : m_op(&op), m_c(c) { }
		btod_additive<N> &get_op() { return *m_op; }
		double get_coeff() const { return m_c; }
	} node_t;

private:
	std::list<node_t> m_ops; //!< List of operations
	block_index_space<N> m_bis; //!< Block index space
	symmetry<N, double> m_sym; //!< Symmetry of operation
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the base operation
		\param op Operation.
		\param c Coefficient.
	 **/
	btod_sum(btod_additive<N> &op, double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_sum();

	//@}


	//!	\name Implementation of libtensor::direct_tensor_operation<N>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	virtual const symmetry<N, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<N, double> &bt) throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx)
		throw(exception);
	//@}


	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual const assignment_schedule<N, double> &get_schedule();
	virtual void compute_block(tensor_i<N, double> &blk,
		const index<N> &i) { }
	virtual void compute_block(tensor_i<N, double> &blk, const index<N> &i,
		double c) { }
	virtual void perform(block_tensor_i<N, double> &bt, double c)
		throw(exception);
	virtual void perform(block_tensor_i<N, double> &bt, const index<N> &idx,
		double c) throw(exception);
	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Adds an operation to the sequence
		\param op Operation.
		\param c Coefficient.
	 **/
	void add_op(btod_additive<N> &op, double c = 1.0);

	//@}

private:
	btod_sum<N> &operator=(const btod_sum<N>&);

};


template<size_t N>
const char* btod_sum<N>::k_clazz = "btod_sum<N>";


template<size_t N>
inline btod_sum<N>::btod_sum(btod_additive<N> &op, double c) :
	m_bis(op.get_bis()), m_sym(op.get_bis()) {

	add_op(op, c);
}


template<size_t N>
btod_sum<N>::~btod_sum() {

}


template<size_t N>
inline const block_index_space<N> &btod_sum<N>::get_bis() const {

	return m_bis;
}


template<size_t N>
const symmetry<N, double> &btod_sum<N>::get_symmetry() const {

	return m_sym;
}


template<size_t N>
const assignment_schedule<N, double> &btod_sum<N>::get_schedule() {

	throw not_implemented(g_ns, k_clazz, "get_schedule()",
		__FILE__, __LINE__);
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	timings< btod_sum<N> >::start_timer();

	block_tensor_ctrl<N, double> ctrl(bt);
	ctrl.req_zero_all_blocks();

	typename std::list<node_t>::iterator i = m_ops.begin();
	for(; i != m_ops.end(); i++) {
		i->get_op().perform(bt, i->get_coeff());
	}

	timings< btod_sum<N> >::stop_timer();
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
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	timings< btod_sum<N> >::start_timer();

	typename std::list<node_t>::iterator i = m_ops.begin();
	for(; i != m_ops.end(); i++) {
		i->get_op().perform(bt, i->get_coeff() * c);
	}

	timings< btod_sum<N> >::stop_timer();
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx,
	double c) throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, const index<N>&, double)";
	
	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N>
void btod_sum<N>::add_op(btod_additive<N> &op, double c) {

	static const char *method = "add_op(btod_additive<N>&, double)";

	if(!op.get_bis().equals(m_bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incompatible block index space.");
	}

	m_ops.push_back(node_t(op, c));
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_H

