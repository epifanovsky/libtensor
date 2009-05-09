#ifndef LIBTENSOR_CONTRACTION2_PROCESSOR_H
#define LIBTENSOR_CONTRACTION2_PROCESSOR_H

#include "defs.h"
#include "exception.h"
#include "contraction2_list.h"

namespace libtensor {

template<size_t N>
class contraction2_processor {
private:
	contraction2_list<N> &m_list; //!< Contraction list
	double *m_ptrc; //!< Result data pointer
	const double *m_ptra; //!< First argument data pointer
	const double *m_ptrb; //!< Second argument data pointer

public:
	contraction2_processor(contraction2_list<N> &list, double *ptrc,
		const double *ptra, const double *ptrb);

	void contract() throw(exception);

private:
	void nodefn_loop(const contraction2_node &node);
	void nodefn_loop_mult(const contraction2_node &node);
};

template<size_t N>
contraction2_processor<N>::contraction2_processor(contraction2_list<N> &list,
	double *ptrc, const double *ptra, const double *ptrb) :
	m_list(list), m_ptrc(ptrc), m_ptra(ptra), m_ptrb(ptrb) {
}

template<size_t N>
void contraction2_processor<N>::contract() throw(exception) {
	if (m_list.get_length() == 0)
		return;

	size_t lasta, lastb, lastc;
	size_t node = m_list.get_first();
	for(size_t i = 0; i < m_list.get_length(); i++) {
		if(m_list.get_node(node).m_inca == 1) lasta = node;
		if(m_list.get_node(node).m_incb == 1) lastb = node;
		if(m_list.get_node(node).m_incc == 1) lastc = node;
		node = m_list.get_next(node);
	}

	if(lasta == lastb) {

	} else if(lasta == lastc) {

	} else if(lastb == lastc) {

	} else {

	}
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_PROCESSOR_H
