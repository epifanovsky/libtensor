#include "tod_sum.h"

namespace libtensor {

tod_sum::tod_sum() : m_head(NULL), m_tail(NULL) {
}

tod_sum::~tod_sum() {
	struct list_node *node = m_head;
	m_head = NULL; m_tail = NULL;
	while(node != NULL) {
		struct list_node *next = node->m_next;
		delete node; node = next;
	}
}

void tod_sum::prefetch() throw(exception) {
}

void tod_sum::perform(tensor_i<double> &t) throw(exception) {
}

void tod_sum::add_op(tod_additive &op, double c) throw(exception) {
	struct list_node *node = new struct list_node(op, c);
	if(m_tail == NULL) {
		m_head = node; m_tail = node;
	} else {
		m_tail->m_next = node;
		m_tail = node;
	}
}

}

