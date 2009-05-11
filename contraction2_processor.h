#ifndef LIBTENSOR_CONTRACTION2_PROCESSOR_H
#define LIBTENSOR_CONTRACTION2_PROCESSOR_H

#include "defs.h"
#include "exception.h"
#include "contraction2_list.h"

namespace libtensor {

/**	\brief Computes the contraction of two tensors

	\ingroup libtensor
 **/
template<size_t N>
class contraction2_processor {
private:
	typedef void (contraction2_processor<N>::*nodefn_t)(size_t);

private:
	contraction2_list<N> &m_list; //!< Contraction list
	double *m_ptrc; //!< Result data pointer
	const double *m_ptra; //!< First argument data pointer
	const double *m_ptrb; //!< Second argument data pointer
	double *m_reg_ptrc;
	const double *m_reg_ptra;
	const double *m_reg_ptrb;
	size_t m_num_nodes;
	size_t m_nodes[N];
	nodefn_t m_funcs[N];

public:
	contraction2_processor(contraction2_list<N> &list, double *ptrc,
		const double *ptra, const double *ptrb);

	void contract() throw(exception);

private:
	void exec_next_node(size_t cur_node);
	void nodefn_loop(size_t node);
	void nodefn_ddot(size_t node);
	void nodefn_dgemv_a(size_t node);
	void nodefn_dgemv_b(size_t node);
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

	size_t lasta, lastb;
	size_t node = m_list.get_first();
	for(size_t i = 0; i < m_list.get_length(); i++) {
		if(m_list.get_node(node).m_inca == 1) lasta = node;
		if(m_list.get_node(node).m_incb == 1) lastb = node;
		node = m_list.get_next(node);
	}

	size_t skip1, skip2, skip3;
	m_num_nodes = 0;

	if(lasta == lastb) {
		skip1 = skip2 = skip3 = lasta;

		// Same last index in a and b
		m_nodes[m_num_nodes] = lasta;
		m_funcs[m_num_nodes] = &contraction2_processor<N>::nodefn_ddot;
		m_num_nodes++;

		bool dgemv_a = false, dgemv_b = false;

		// Find last index in c
		node = m_list.get_first();
		for(size_t i = 0; i<m_list.get_length(); i++) {
			if(m_list.get_node(node).m_incc == 1) {
				if(m_list.get_node(node).m_inca != 0) dgemv_a = true;
				else dgemv_b = true;
				break;
			}
			node = m_list.get_next(node);
		}
		if(dgemv_a) {
			skip2 = node;
			m_nodes[m_num_nodes] = node;
			m_funcs[m_num_nodes] = &contraction2_processor<N>::nodefn_dgemv_a;
			m_num_nodes++;
		}
		if(dgemv_b) {
			skip2 = node;
			m_nodes[m_num_nodes] = node;
			m_funcs[m_num_nodes] = &contraction2_processor<N>::nodefn_dgemv_b;
			m_num_nodes++;
/*
			size_t seek_incc = m_list.get_node(node).m_weight;
			node = m_list.get_first();
			for(size_t i = 0; i<m_list.get_length(); i++) {
				if(m_list.get_node(node).m_incc == seek_incc) {
				}
				node = m_list.get_next(node);
			}
			*/
		}


		node = m_list.get_first();
		for(size_t i = 0; i<m_list.get_length(); i++) {
			if(node != skip1 && node != skip2 && node != skip3) {
				m_nodes[m_num_nodes] = node;
				m_funcs[m_num_nodes] = &contraction2_processor<N>::nodefn_loop;
				m_num_nodes++;
			}
			node = m_list.get_next(node);
		}
	} else {
		throw_exc("contraction2_processor<N>", "contract()",
			"Not implemented yet");
	}

	m_reg_ptrc = m_ptrc;
	m_reg_ptra = m_ptra;
	m_reg_ptrb = m_ptrb;

	exec_next_node(m_num_nodes);
}

template<size_t N>
inline void contraction2_processor<N>::exec_next_node(size_t cur_node) {
	if(cur_node == 0) return;
	(this->*(m_funcs[cur_node-1]))(cur_node-1);
}

template<size_t N>
void contraction2_processor<N>::nodefn_loop(size_t node) {
	const contraction2_node &contr(m_list.get_node(m_nodes[node]));
	double *ptrc = m_reg_ptrc;
	const double *ptra = m_reg_ptra, *ptrb = m_reg_ptrb;
	for(size_t i=0; i<contr.m_weight; i++) {
		m_reg_ptrc = ptrc;
		m_reg_ptra = ptra;
		m_reg_ptrb = ptrb;
		exec_next_node(node);
		ptra += contr.m_inca;
		ptrb += contr.m_incb;
		ptrc += contr.m_incc;
	}
}

template<size_t N>
void contraction2_processor<N>::nodefn_ddot(size_t node) {
	const contraction2_node &contr(m_list.get_node(m_nodes[node]));
	*m_reg_ptrc = cblas_ddot(contr.m_weight, m_reg_ptra, 1, m_reg_ptrb, 1);
}

template<size_t N>
void contraction2_processor<N>::nodefn_dgemv_a(size_t node) {
	const contraction2_node &contr(m_list.get_node(m_nodes[node]));
	const contraction2_node &contr_prev(m_list.get_node(m_nodes[node-1]));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, contr.m_weight,
		contr_prev.m_weight, 1.0, m_reg_ptra, contr.m_inca,
		m_reg_ptrb, 1, 0.0, m_reg_ptrc, 1);
}

template<size_t N>
void contraction2_processor<N>::nodefn_dgemv_b(size_t node) {
	const contraction2_node &contr(m_list.get_node(m_nodes[node]));
	const contraction2_node &contr_prev(m_list.get_node(m_nodes[node-1]));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, contr.m_weight,
		contr_prev.m_weight, 1.0, m_reg_ptrb, contr.m_incb,
		m_reg_ptra, 1, 0.0, m_reg_ptrc, 1);
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_PROCESSOR_H
