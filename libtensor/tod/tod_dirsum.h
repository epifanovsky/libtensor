#ifndef LIBTENSOR_TOD_DIRSUM_H
#define LIBTENSOR_TOD_DIRSUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "bad_dimensions.h"
#include "contraction2.h"
#include "contraction2_list_builder.h"

namespace libtensor {


/**	\brief Computes the direct sum of two tensors
	\tparam N Order of the first %tensor.
	\tparam M Order of the second %tensor.

	Given two tensors \f$ a_{ij\cdots} \f$ and \f$ b_{mn\cdots} \f$,
	the operation computes
	\f$ c_{ij\cdots mn\cdots} = k_a a_{ij\cdots} + k_b b_{mn\cdots} \f$.

	The order of %tensor indexes in the result can be specified using
	a permutation.

	\ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_dirsum : public timings< tod_dirsum<N, M> > {
public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	struct registers {
		const double *m_ptra;
		const double *m_ptrb;
		double *m_ptrc;
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef typename loop_list_t::iterator loop_list_iterator_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incb, m_incc;
		void (tod_dirsum<N, M>::*m_fn)(registers &);
		loop_list_node() : m_weight(0), m_inca(0), m_incb(0),
			m_incc(0), m_fn(0) { }
		loop_list_node(size_t weight, size_t inca, size_t incb,
			size_t incc) : m_weight(weight), m_inca(inca),
			m_incb(incb), m_incc(incc), m_fn(0) { }
	};

	//!	c_ji = a_i b_j
	struct {
		double m_kc;
		size_t m_n;
		size_t m_stepb;
		size_t m_stepc;
	} m_add_a;

	class loop_list_adapter {
	private:
		loop_list_t &m_list;

	public:
		loop_list_adapter(loop_list_t &list) : m_list(list) { }

		void append(size_t weight, size_t inca, size_t incb,
			size_t incc) {

			m_list.push_back(loop_list_node(
				weight, inca, incb, incc));
		}
	};


private:
	tensor_i<k_ordera, double> &m_ta; //!< First %tensor (A)
	tensor_i<k_orderb, double> &m_tb; //!< Second %tensor (B)
	double m_ka; //!< Coefficient A
	double m_kb; //!< Coefficient B
	permutation<k_orderc> m_permc; //!< Permutation of the result
	dimensions<k_orderc> m_dimsc; //!< Dimensions of the result
	loop_list_t m_list; //!< Loop list

public:
	/**	\brief Initializes the operation
	 **/
	tod_dirsum(tensor_i<k_ordera, double> &ta, double ka,
		tensor_i<k_orderb, double> &tb, double kb) :
		m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb),
		m_dimsc(mk_dimsc(ta, tb)) { }

	/**	\brief Initializes the operation
	 **/
	tod_dirsum(tensor_i<k_ordera, double> &ta, double ka,
		tensor_i<k_orderb, double> &tb, double kb,
		const permutation<k_orderc> &permc) :
		m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb), m_permc(permc),
		m_dimsc(mk_dimsc(ta, tb)) {

		m_dimsc.permute(m_permc);
	}

	/**	\brief Performs the operation
	 **/
	void perform(tensor_i<k_orderc, double> &tc);

	/**	\brief Performs the operation (additive)
	 **/
	void perform(tensor_i<k_orderc, double> &tc, double kc);

private:
	static dimensions<N + M> mk_dimsc(tensor_i<k_ordera, double> &ta,
		tensor_i<k_orderb, double> &tb);
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double kc);

private:
	void exec(loop_list_iterator_t &i, registers &regs);
	void fn_loop(loop_list_iterator_t &i, registers &regs);
	void fn_add_a(registers &regs);

};


template<size_t N, size_t M>
const char *tod_dirsum<N, M>::k_clazz = "tod_dirsum<N, M>";


template<size_t N, size_t M>
void tod_dirsum<N, M>::perform(tensor_i<k_orderc, double> &tc) {

	static const char *method = "perform(tensor_i<N + M, double>&)";

	if(!m_dimsc.equals(tc.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method,
			__FILE__, __LINE__, "tc");
	}

	do_perform(tc, true, 1.0);
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::perform(tensor_i<k_orderc, double> &tc, double kc) {

	static const char *method =
		"perform(tensor_i<N + M, double>&, double)";

	if(!m_dimsc.equals(tc.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method,
			__FILE__, __LINE__, "tc");
	}

	do_perform(tc, false, kc);
}


template<size_t N, size_t M>
dimensions<N + M> tod_dirsum<N, M>::mk_dimsc(
	tensor_i<k_ordera, double> &ta, tensor_i<k_orderb, double> &tb) {

	const dimensions<k_ordera> &dimsa = ta.get_dims();
	const dimensions<k_orderb> &dimsb = tb.get_dims();

	index<k_orderc> i1, i2;
	for(register size_t i = 0; i < k_ordera; i++)
		i2[i] = dimsa[i] - 1;
	for(register size_t i = 0; i < k_orderb; i++)
		i2[k_ordera + i] = dimsb[i] - 1;

	return dimensions<k_orderc>(index_range<k_orderc>(i1, i2));
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::do_perform(tensor_i<k_orderc, double> &tc, bool zero,
	double kc) {

	tod_dirsum<N, M>::start_timer();

	//	We use contraction2 here to build a loop list
	//
	contraction2<N, M, 0> contr(m_permc);
	loop_list_adapter list_adapter(m_list);
	contraction2_list_builder<N, M, 0, loop_list_adapter> lstbld(contr);
	lstbld.populate(list_adapter, m_ta.get_dims(), m_tb.get_dims(),
		tc.get_dims());

	tensor_ctrl<k_ordera, double> ctrla(m_ta);
	tensor_ctrl<k_orderb, double> ctrlb(m_tb);
	tensor_ctrl<k_orderc, double> ctrlc(tc);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	//	Zero the output tensor if necessary
	//
	if(zero) {
		tod_dirsum<N, M>::start_timer("zero");
		size_t szc = tc.get_dims().get_size();
		for(size_t i = 0; i < szc; i++) ptrc[i] = 0.0;
		tod_dirsum<N, M>::stop_timer("zero");
	}

	//	Install the kernel on the fastest-running index in A
	//
	loop_list_iterator_t i1 = m_list.begin();
	while(i1 != m_list.end() && i1->m_inca != 1) i1++;
	if(i1 != m_list.end()) {
		i1->m_fn = &tod_dirsum<N, M>::fn_add_a;
		m_add_a.m_kc = kc;
		m_add_a.m_n = i1->m_weight;
		m_add_a.m_stepb = i1->m_incb;
		m_add_a.m_stepc = i1->m_incc;
		m_list.splice(m_list.end(), m_list, i1);
	}

	//	Run the loop
	//
	try {
		registers regs;
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		regs.m_ptrc = ptrc;

		loop_list_iterator_t i = m_list.begin();
		if(i != m_list.end()) exec(i, regs);
	} catch(exception&) {
		tod_dirsum<N, M>::stop_timer();
		throw;
	}

	ctrla.ret_dataptr(ptra);
	ctrlb.ret_dataptr(ptrb);
	ctrlc.ret_dataptr(ptrc);

	tod_dirsum<N, M>::stop_timer();
}


template<size_t N, size_t M>
inline void tod_dirsum<N, M>::exec(loop_list_iterator_t &i, registers &r) {

	void (tod_dirsum<N, M>::*fnptr)(registers&) = i->m_fn;

	if(fnptr == 0) fn_loop(i, r);
	else (this->*fnptr)(r);
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::fn_loop(loop_list_iterator_t &i, registers &r) {

	loop_list_iterator_t j = i; j++;
	if(j == m_list.end()) return;

	const double *ptra = r.m_ptra, *ptrb = r.m_ptrb;
	double *ptrc = r.m_ptrc;

	for(size_t k = 0; k < i->m_weight; k++) {

		r.m_ptra = ptra;
		r.m_ptrb = ptrb;
		r.m_ptrc = ptrc;
		exec(j, r);
		ptra += i->m_inca;
		ptrb += i->m_incb;
		ptrc += i->m_incc;
	}
}


template<size_t N, size_t M>
void tod_dirsum<N, M>::fn_add_a(registers &r) {

	const double *ptra = r.m_ptra, *ptrb = r.m_ptrb;
	double *ptrc = r.m_ptrc;

	double kc = m_add_a.m_kc;

	for(size_t k = 0; k < m_add_a.m_n; k++) {
		ptrc[0] += kc * (m_ka * ptra[k] + m_kb * ptrb[0]);
		ptrb += m_add_a.m_stepb;
		ptrc += m_add_a.m_stepc;
	}
}


} // namespace libtensor

#endif // LIBTENOSR_TOD_DIRSUM_H