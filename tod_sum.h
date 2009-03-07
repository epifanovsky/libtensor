#ifndef LIBTENSOR_TOD_SUM_H
#define LIBTENSOR_TOD_SUM_H

#include "defs.h"
#include "exception.h"
#include "direct_tensor_operation.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Adds results of a sequence of operations (double)

	\ingroup libtensor
**/
class tod_sum : public direct_tensor_operation<double> {
private:
	struct list_node {
		tod_additive &m_op;
		double m_c;
		struct list_node *m_next;
		list_node(tod_additive &op, double c);
	};

	struct list_node *m_head;
	struct list_node *m_tail;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor
	**/
	tod_sum();

	/**	\brief Virtual destructor
	**/
	virtual ~tod_sum();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	virtual void perform(tensor_i<double> &t) throw(exception);
	//@}

	/**	\brief Adds an operation to the sequence
	**/
	void add_op(tod_additive &op, double c) throw(exception);
};

inline tod_sum::list_node::list_node(tod_additive &op, double c) :
	m_op(op), m_c(c), m_next(NULL) {
}

}

#endif // LIBTENSOR_TOD_SUM_H

