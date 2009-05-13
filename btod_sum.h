#ifndef LIBTENSOR_BTOD_SUM_H
#define LIBTENSOR_BTOD_SUM_H

#include "defs.h"
#include "exception.h"
#include "btod_additive.h"

namespace libtensor {

/**	\brief Adds results of a sequence of operations on block_tensors (double)

	\ingroup libtensor
**/
template<size_t N>
class btod_sum : public direct_block_tensor_operation<N,double> {
private:
	struct list_node {
		btod_additive<N> &m_op;
		double m_c;
		struct list_node *m_next;
		list_node(btod_additive<N> &op, double c);
	};

	direct_block_tensor_operation<N,double> &m_baseop; //!< Base operation
	struct list_node *m_head; //!< Head of the list of additional operations
	struct list_node *m_tail; //!< Tail of the list of additional operations

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor
	**/
	btod_sum(direct_block_tensor_operation<N,double> &op);

	/**	\brief Virtual destructor
	**/
	virtual ~btod_sum();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void perform(block_tensor_i<N,double> &bt) throw(exception);
	//@}

	/**	\brief Adds an operation to the sequence
	**/
	void add_op(btod_additive<N> &op, double c) throw(exception);
};

template<size_t N>
inline btod_sum<N>::btod_sum(direct_block_tensor_operation<N,double> &op) :
	m_baseop(op), m_head(NULL), m_tail(NULL) {
}

template<size_t N>
btod_sum<N>::~btod_sum() {
	struct list_node *node = m_head;
	m_head = NULL; m_tail = NULL;
	while(node != NULL) {
		struct list_node *next = node->m_next;
		delete node; 
		node = next;
	}
}

template<size_t N>
void btod_sum<N>::perform(block_tensor_i<N,double> &bt) throw(exception) {
	m_baseop.perform(bt);
	struct list_node *node = m_head;
	while(node != NULL) {
		node->m_op.perform(bt, node->m_c);
		node = node->m_next;
	}
}

template<size_t N>
void btod_sum<N>::add_op(btod_additive<N> &op, double c) throw(exception) {
	struct list_node *node = new struct list_node(op, c);
	if(m_tail == NULL) {
		m_head = node; m_tail = node;
	} else {
		m_tail->m_next = node;
		m_tail = node;
	}
}

template<size_t N>
inline btod_sum<N>::list_node::list_node(btod_additive<N> &op, double c) :
	m_op(op), m_c(c), m_next(NULL) {
}

}

#endif // LIBTENSOR_TOD_SUM_H

