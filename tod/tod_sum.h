#ifndef LIBTENSOR_TOD_SUM_H
#define LIBTENSOR_TOD_SUM_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Adds results of a sequence of operations (double)

	\ingroup libtensor
**/
template<size_t N>
class tod_sum {
private:
	struct list_node {
		tod_additive<N> &m_op;
		double m_c;
		struct list_node *m_next;
		list_node(tod_additive<N> &op, double c);
	};

	tod_additive<N> &m_baseop; //!< Base operation
	struct list_node *m_head; //!< Head of the list of additional operations
	struct list_node *m_tail; //!< Tail of the list of additional operations

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor
	**/
	tod_sum(tod_additive<N> &op);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_sum();

	//@}

	void prefetch() throw(exception);
	void perform(tensor_i<N,double> &t) throw(exception);

	/**	\brief Adds an operation to the sequence
	**/
	void add_op(tod_additive<N> &op, double c) throw(exception);
};

template<size_t N>
inline tod_sum<N>::tod_sum(tod_additive<N> &op) :
	m_baseop(op), m_head(NULL), m_tail(NULL) {
}

template<size_t N>
tod_sum<N>::~tod_sum() {
	struct list_node *node = m_head;
	m_head = NULL; m_tail = NULL;
	while(node != NULL) {
		struct list_node *next = node->m_next;
		delete node; node = next;
	}
}

template<size_t N>
void tod_sum<N>::prefetch() throw(exception) {
	m_baseop.prefetch();
	struct list_node *node = m_head;
	while(node != NULL) {
		node->m_op.prefetch();
		node = node->m_next;
	}
}

template<size_t N>
void tod_sum<N>::perform(tensor_i<N,double> &t) throw(exception) {
	m_baseop.perform(t);
	struct list_node *node = m_head;
	while(node != NULL) {
		node->m_op.perform(t, node->m_c);
		node = node->m_next;
	}
}

template<size_t N>
void tod_sum<N>::add_op(tod_additive<N> &op, double c) throw(exception) {
	struct list_node *node = new struct list_node(op, c);
	if(m_tail == NULL) {
		m_head = node; m_tail = node;
	} else {
		m_tail->m_next = node;
		m_tail = node;
	}
}

template<size_t N>
inline tod_sum<N>::list_node::list_node(tod_additive<N> &op, double c) :
	m_op(op), m_c(c), m_next(NULL) {
}

}

#endif // LIBTENSOR_TOD_SUM_H

