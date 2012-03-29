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

	virtual void sync_on();
	virtual void sync_off();

	//@}


	//!	\name Implementation of libtensor::additive_btod<N>
	//@{

	virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
	    const index<N> &i, const tensor_transf<N, double> &tr, double c,
	    cpu_pool &cpus);
	virtual void perform(block_tensor_i<N, double> &bt);
	virtual void perform(block_tensor_i<N, double> &bt, double c);

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


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_sum_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_SUM_H

