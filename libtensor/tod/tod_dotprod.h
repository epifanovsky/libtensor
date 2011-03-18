#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../linalg/linalg.h"
#include "../core/permutation.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "contraction2.h"
#include "bad_dimensions.h"
#include "processor.h"

namespace libtensor {

/**	\brief Calculates the dot product of two tensors
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_dotprod : public timings<tod_dotprod<N> > {
public:
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
		size_t m_n, m_inca, m_incb;
	public:
		op_ddot(size_t n, size_t inca, size_t incb) :
			m_n(n), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);

		static const char *k_clazz;
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

	/**	\brief Prefetches the arguments
	 **/
	void prefetch();

	/**	\brief Computes the dot product
	 **/
	double calculate();

private:
	bool verify_dims();
	void clean_list();
	void build_list( loop_list_t &list, const dimensions<N> &da,
		const permutation<N> &pa, const dimensions<N> &db ) throw(out_of_memory);
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
		bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t1, t2.");
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
		bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t1, t2.");
	}
}

template<size_t N>
void tod_dotprod<N>::prefetch() {

	m_tctrl1.req_prefetch();
	m_tctrl2.req_prefetch();
}

template<size_t N>
double tod_dotprod<N>::calculate() {
	tod_dotprod<N>::start_timer();

	permutation<N> perma(m_perm1);
	perma.invert();
	permutation<N> permb(m_perm2);
	permb.permute(perma);

	const dimensions<N> &dima(m_t1.get_dims());
	const dimensions<N> &dimb(m_t2.get_dims());

	build_list(m_list,dimb,permb,dima);

	const double *pb = m_tctrl1.req_const_dataptr();
	const double *pa = m_tctrl2.req_const_dataptr();

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
	m_tctrl1.ret_dataptr(pb);
	m_tctrl2.ret_dataptr(pa);

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
void tod_dotprod<N>::build_list(loop_list_t &list, const dimensions<N> &da,
	const permutation<N> &pa, const dimensions<N> &db)
	throw(out_of_memory) {

	sequence<N, size_t> ia(0);
	for(size_t i = 0; i < N; i++) ia[i] = i;
	pa.apply(ia);

	// loop over all indices and build the list
	size_t pos = 0;
	try {
		typename loop_list_t::iterator posa = list.end(),
			posb = list.end();
		while(pos < N) {
			size_t len = 1;
			size_t iapos = ia[pos];
			while(pos < N) {
				if(ia[pos] != iapos) break;
				len *= da.get_dim(iapos);
				pos++; iapos++;
			}

			size_t inca = da.get_increment(iapos - 1);
			size_t incb = db.get_increment(pos - 1);

			typename loop_list_t::iterator it = list.insert(
				list.end(),loop_list_node(len,inca,incb));

			// we know that in the last loop incb=1 !!!
			if(inca == 1) {
				if(inca == incb) {
					it->m_op = new op_ddot(len, inca, incb);
				} else {
					posa = it;
				}
			} else {
				if(incb == 1) posb = it;
				else it->m_op = new op_loop(len, inca, incb);
			}
		}

		if(posa != posb) {
			if(posa->m_weight > posb->m_weight) {
				posa->m_op = new op_ddot(posa->m_weight,
					posa->m_inca, posa->m_incb);
				posb->m_op = new op_loop(posb->m_weight,
					posb->m_inca, posb->m_incb);
				list.splice(list.end(), list, posa);
			} else {
				posa->m_op = new op_loop(posa->m_weight,
					posa->m_inca, posa->m_incb);
				posb->m_op = new op_ddot(posb->m_weight,
					posb->m_inca, posb->m_incb);
				list.splice(posb, list, posa);
			}
		}

	} catch(std::bad_alloc &e) {

		clean_list();
		throw out_of_memory("libtensor",k_clazz,
			"build_list(loop_list_t&,const dimensions<N>&,"
			"const permutation<N>&,const dimensions<N>&)",
			__FILE__,__LINE__,e.what());
	}
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

	*(regs.m_ptrc) += linalg::x_p_p(m_n, regs.m_ptra, m_inca,
		regs.m_ptrb, m_incb);

	tod_dotprod<N>::op_ddot::stop_timer();

}

} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_H
