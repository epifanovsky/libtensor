#ifndef LIBTENSOR_TOD_SCATTER_H
#define LIBTENSOR_TOD_SCATTER_H

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


/**	\brief Scatters a lower-order %tensor in a higher-order %tensor
	\tparam N Order of the first %tensor.
	\tparam M Order of the result less the order of the first %tensor.

	Given a %tensor \f$ a_{ij\cdots} \f$, the operation computes
	\f$ c_{\cdots ij\cdots} = k_a a_{ij\cdots} \f$.

	The order of %tensor indexes in the result can be specified using
	a permutation.

	\sa tod_dirsum

	\ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_scatter : public timings< tod_scatter<N, M> > {
public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	struct registers {
		const double *m_ptra;
		double *m_ptrc;
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef typename loop_list_t::iterator loop_list_iterator_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incc;
		void (tod_scatter<N, M>::*m_fn)(registers &);
		loop_list_node() : m_weight(0), m_inca(0), m_incc(0), m_fn(0)
			{ }
		loop_list_node(size_t weight, size_t inca, size_t incc) :
			m_weight(weight), m_inca(inca), m_incc(incc), m_fn(0)
			{ }
	};

	//!	c_ji = a_i
	struct {
		double m_kc;
		size_t m_n;
		size_t m_stepc;
	} m_scatter;

private:
	tensor_i<k_ordera, double> &m_ta; //!< First %tensor (A)
	double m_ka; //!< Coefficient A
	permutation<k_orderc> m_permc; //!< Permutation of the result
	loop_list_t m_list; //!< Loop list

public:
	/**	\brief Initializes the operation
	 **/
	tod_scatter(tensor_i<k_ordera, double> &ta, double ka) :
		m_ta(ta), m_ka(ka) { }

	/**	\brief Initializes the operation
	 **/
	tod_scatter(tensor_i<k_ordera, double> &ta, double ka,
		const permutation<k_orderc> &permc) :
		m_ta(ta), m_ka(ka), m_permc(permc) { }

	/**	\brief Performs the operation
	 **/
	void perform(tensor_i<k_orderc, double> &tc);

	/**	\brief Performs the operation (additive)
	 **/
	void perform(tensor_i<k_orderc, double> &tc, double kc);

private:
	void check_dimsc(tensor_i<k_orderc, double> &tc);
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double kc);

private:
	void exec(loop_list_iterator_t &i, registers &regs);
	void fn_loop(loop_list_iterator_t &i, registers &regs);
	void fn_scatter(registers &regs);

};


template<size_t N, size_t M>
const char *tod_scatter<N, M>::k_clazz = "tod_scatter<N, M>";


template<size_t N, size_t M>
void tod_scatter<N, M>::perform(tensor_i<k_orderc, double> &tc) {

	check_dimsc(tc);
	do_perform(tc, true, 1.0);
}


template<size_t N, size_t M>
void tod_scatter<N, M>::perform(tensor_i<k_orderc, double> &tc, double kc) {

	check_dimsc(tc);
	do_perform(tc, false, kc);
}


template<size_t N, size_t M>
void tod_scatter<N, M>::check_dimsc(tensor_i<k_orderc, double> &tc) {

	static const char *method = "check_dimsc(tensor_i<N + M, double>&)";

	permutation<k_orderc> pinv(m_permc, true);
	dimensions<k_orderc> dimsc(tc.get_dims());
	dimsc.permute(pinv);

	bool bad_dims = false;
	const dimensions<k_ordera> &dimsa = m_ta.get_dims();
	for(size_t i = 0; i < k_ordera; i++) {
		if(dimsc[k_orderc - k_ordera + i] != dimsa[i]) {
			bad_dims = true;
			break;
		}
	}
	if(bad_dims) {
		throw bad_dimensions(g_ns, k_clazz, method,
			__FILE__, __LINE__, "tc");
	}
}


template<size_t N, size_t M>
void tod_scatter<N, M>::do_perform(tensor_i<k_orderc, double> &tc, bool zero,
	double kc) {

	tod_scatter<N, M>::start_timer();

	size_t seq[k_orderc];
	for(size_t i = 0; i < k_orderc - k_ordera; i++) seq[i] = k_ordera;
	for(size_t i = 0; i < k_ordera; i++)
		seq[k_orderc - k_ordera + i] = i;
	m_permc.apply(seq);

	const dimensions<k_ordera> &dimsa = m_ta.get_dims();
	const dimensions<k_orderc> &dimsc = tc.get_dims();
	m_list.clear();
	for(size_t i = 0; i < k_orderc; i++) {
		if(seq[i] == k_ordera) {
			m_list.push_back(loop_list_node(
				dimsc[i], 0, dimsc.get_increment(i)));
		} else {
			m_list.push_back(loop_list_node(
				dimsc[i], dimsa.get_increment(seq[i]),
				dimsc.get_increment(i)));
		}
	}

	tensor_ctrl<k_ordera, double> ctrla(m_ta);
	tensor_ctrl<k_orderc, double> ctrlc(tc);

	const double *ptra = ctrla.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	//	Zero the output tensor if necessary
	//
	if(zero) {
		tod_scatter<N, M>::start_timer("zero");
		size_t szc = dimsc.get_size();
		for(size_t i = 0; i < szc; i++) ptrc[i] = 0.0;
		tod_scatter<N, M>::stop_timer("zero");
	}

	//	Install the kernel on the fastest-running index in A
	//
	loop_list_iterator_t i1 = m_list.begin();
	while(i1 != m_list.end() && i1->m_inca != 1) i1++;
	if(i1 != m_list.end()) {
		i1->m_fn = &tod_scatter<N, M>::fn_scatter;
		m_scatter.m_kc = kc;
		m_scatter.m_n = i1->m_weight;
		m_scatter.m_stepc = i1->m_incc;
		m_list.splice(m_list.end(), m_list, i1);
	}

	//	Run the loop
	//
	try {
		registers regs;
		regs.m_ptra = ptra;
		regs.m_ptrc = ptrc;

		loop_list_iterator_t i = m_list.begin();
		if(i != m_list.end()) exec(i, regs);
	} catch(exception&) {
		tod_scatter<N, M>::stop_timer();
		throw;
	}

	ctrla.ret_dataptr(ptra);
	ctrlc.ret_dataptr(ptrc);

	tod_scatter<N, M>::stop_timer();
}


template<size_t N, size_t M>
inline void tod_scatter<N, M>::exec(loop_list_iterator_t &i, registers &r) {

	void (tod_scatter<N, M>::*fnptr)(registers&) = i->m_fn;

	if(fnptr == 0) fn_loop(i, r);
	else (this->*fnptr)(r);
}


template<size_t N, size_t M>
void tod_scatter<N, M>::fn_loop(loop_list_iterator_t &i, registers &r) {

	loop_list_iterator_t j = i; j++;
	if(j == m_list.end()) return;

	const double *ptra = r.m_ptra;
	double *ptrc = r.m_ptrc;

	for(size_t k = 0; k < i->m_weight; k++) {

		r.m_ptra = ptra;
		r.m_ptrc = ptrc;
		exec(j, r);
		ptra += i->m_inca;
		ptrc += i->m_incc;
	}
}


template<size_t N, size_t M>
void tod_scatter<N, M>::fn_scatter(registers &r) {

	const double *ptra = r.m_ptra;
	double *ptrc = r.m_ptrc;

	double kc = m_scatter.m_kc * m_ka;

	for(size_t k = 0; k < m_scatter.m_n; k++) {
		ptrc[0] += kc * ptra[k];
		ptrc += m_scatter.m_stepc;
	}
}


} // namespace libtensor

#endif // LIBTENOSR_TOD_SCATTER_H

