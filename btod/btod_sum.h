#ifndef LIBTENSOR_BTOD_SUM_H
#define LIBTENSOR_BTOD_SUM_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "btod_additive.h"

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

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the base operation
		\param op Operation.
		\param c Coefficient.
	 **/
	btod_sum(btod_additive<N> &op, double c);

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
	virtual void perform(block_tensor_i<N, double> &bt, double c)
		throw(exception);
	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Adds an operation to the sequence
		\param op Operation.
		\param c Coefficient.
	 **/
	void add_op(btod_additive<N> &op, double c);

	//@}

private:
	btod_sum<N> &operator=(const btod_sum<N>&);

};


template<size_t N>
const char* btod_sum<N>::k_clazz = "btod_sum<N>";


template<size_t N>
inline btod_sum<N>::btod_sum(btod_additive<N> &op, double c) :
	m_bis(op.get_bis()) {

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
	throw_exc("btod_sum<N>", "get_symmetry()", "Not implemented");
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	timings<btod_sum<N> >::start_timer();

	typename std::list<node_t>::iterator i = m_ops.begin();
	for(; i != m_ops.end(); i++) {
		i->get_op().perform(bt, i->get_coeff());
	}

	timings<btod_sum<N> >::stop_timer();
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, const index<N> &idx)
	throw(exception) {

	throw_exc(k_clazz,
		"perform(block_tensor_i<N, double>&, const index<N>&)", "NIY");
}


template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {

	throw_exc(k_clazz, "perform(block_tensor_i<N, double>&, double)",
		"NIY");
}


template<size_t N>
void btod_sum<N>::add_op(btod_additive<N> &op, double c) {

	static const char *method = "add_op(btod_additive<N>&, double)";

	if(!op.get_bis().equals(m_bis)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incompatible block index space.");
	}

	m_ops.push_back(node_t(op, c));
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_H

