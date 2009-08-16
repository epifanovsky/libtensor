#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "core/permutation.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"
#include "contraction2.h"
#include "processor.h"

namespace libtensor {

/**	\brief Calculates the dot product of two tensors
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_dotprod : public timings<tod_dotprod<N> > {
private:
	friend class timings<tod_dotprod<N> >;
	static const char *k_clazz; //!< Class name

private:
	struct registers {
		const double *m_ptra;
		const double *m_ptrb;
		double *m_ptrc;
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef processor<loop_list_t, registers> processor_t;
	typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incb;
		processor_op_i_t *m_op;
		loop_list_node() : m_weight(0), m_inca(0), m_incb(0),
			m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb)
			: m_weight(weight), m_inca(inca),
			m_incb(incb), m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb) :
			m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_ddot : public processor_op_i_t, public timings<op_ddot> {
	private:
		friend class timings<op_ddot>;
		static const char *k_clazz;
		size_t m_n, m_inca, m_incb;
	public:
		op_ddot(size_t n, size_t inca, size_t incb) :
			m_n(n), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

private:
	tensor_i<N, double> &m_t1; //!< First %tensor
	tensor_ctrl<N, double> m_tctrl1; //!< First %tensor control
	tensor_i<N, double> &m_t2; //!< Second %tensor
	tensor_ctrl<N, double> m_tctrl2; //!< Second %tensor control
	permutation<N> m_perm1; //!< Permutation of the first %tensor
	permutation<N> m_perm2; //!< Permutation of the second %tensor
	loop_list_t m_list; //!< Loop list

public:
	//!	\name Construction and destruction
	//@{

	tod_dotprod(tensor_i<N, double> &t1, tensor_i<N, double> &t2)
		throw(exception);

	tod_dotprod(tensor_i<N, double> &t1, const permutation<N> &perm1,
		tensor_i<N, double> &t2, const permutation<N> &perm2)
		throw(exception);

	//@}

	/**	\brief Computes the dot product
	 **/
	double calculate() throw(exception);

private:
	bool verify_dims();
	void clean_list();
};

template<size_t N>
const char *tod_dotprod<N>::k_clazz = "tod_dotprod<N>";
template<size_t N>
const char *tod_dotprod<N>::op_ddot::k_clazz = "tod_dotprod<N>::op_ddot";

template<size_t N>
tod_dotprod<N>::tod_dotprod(tensor_i<N, double> &t1, tensor_i<N, double> &t2)
	throw(exception)
	: m_t1(t1), m_tctrl1(m_t1), m_t2(t2), m_tctrl2(m_t2) {

	static const char *method = "tod_dotprod(tensor_i<N, double>&, "
		"tensor_i<N, double>&)";

	if(!verify_dims()) {
		throw_exc(k_clazz, method, "Incompatible tensor dimensions");
	}
}

template<size_t N>
tod_dotprod<N>::tod_dotprod(
	tensor_i<N, double> &t1, const permutation<N> &perm1,
	tensor_i<N, double> &t2, const permutation<N> &perm2)
	throw(exception)
	: m_t1(t1), m_tctrl1(m_t1), m_perm1(perm1),
		m_t2(t2), m_tctrl2(m_t2), m_perm2(perm2) {

	static const char *method = "tod_dotprod(tensor_i<N, double>&, "
		"const permutation<N>&, tensor_i<N, double>&, "
		"const permutation<N>&)";

	if(!verify_dims()) {
		throw_exc(k_clazz, method, "Incompatible tensor dimensions");
	}
}

template<size_t N>
double tod_dotprod<N>::calculate() throw(exception) {
	tod_dotprod<N>::start_timer();

	permutation<N> perma(m_perm1);
	perma.invert();
	permutation<N> permb(m_perm2);
	permb.permute(perma);

	size_t idxb[N];
	for(register size_t i = 0; i < N; i++) idxb[i] = i;
	permb.apply(N, idxb);

	const double *pa = m_tctrl1.req_const_dataptr();
	const double *pb = m_tctrl2.req_const_dataptr();

	const dimensions<N> &dima(m_t1.get_dims());
	const dimensions<N> &dimb(m_t2.get_dims());
	for(size_t i = 0; i < N - 1; i++) {
		loop_list_node node(dima[i], dima.get_increment(i),
			dimb.get_increment(idxb[i]));
		node.m_op = new op_loop(dima[i], dima.get_increment(i),
			dimb.get_increment(idxb[i]));
		m_list.push_back(node);
	}
	loop_list_node node(dima[N - 1], dima.get_increment(N - 1),
		dimb.get_increment(idxb[N - 1]));
	node.m_op = new op_ddot(dima[N - 1], dima.get_increment(N - 1),
		dimb.get_increment(idxb[N - 1]));
	m_list.push_back(node);

	double result = 0.0;

	try {
		registers regs;
		regs.m_ptra = pa; regs.m_ptrb = pb; regs.m_ptrc = &result;
		processor_t proc(m_list, regs);
		proc.process_next();
	} catch(exception &e) {
		clean_list();
		throw;
	}

	clean_list();
	m_tctrl1.ret_dataptr(pa);
	m_tctrl2.ret_dataptr(pb);

	tod_dotprod<N>::stop_timer();

	return result;
}

template<size_t N>
bool tod_dotprod<N>::verify_dims() {

	dimensions<N> dims1(m_t1.get_dims());
	dimensions<N> dims2(m_t2.get_dims());
	dims1.permute(m_perm1);
	dims2.permute(m_perm2);
	return dims1.equals(dims2);
}

template<size_t N>
void tod_dotprod<N>::clean_list() {

	for(typename loop_list_t::iterator i = m_list.begin();
		i != m_list.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}
}

template<size_t N>
void tod_dotprod<N>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {

	const double *ptra = regs.m_ptra, *ptrb = regs.m_ptrb;

	for(size_t i = 0; i < m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}
}

template<size_t N>
void tod_dotprod<N>::op_ddot::exec(processor_t &proc, registers &regs)
	throw(exception) {
	tod_dotprod<N>::op_ddot::start_timer();

	*(regs.m_ptrc) += cblas_ddot(
		m_n, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);

	tod_dotprod<N>::op_ddot::stop_timer();

}

} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_H
