#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Makes a copy of a %tensor, scales or permutes %tensor elements
		if necessary
	\tparam N Tensor order.

	This operation makes a scaled and permuted copy of a %tensor.

	<b>Examples</b>

	Plain copy:
	\code
	tensor_i<2, double> &t1(...), &t2(...);
	tod_copy<2> cp(t1);
	cp.perform(t2); // Copies the elements of t1 to t2
	\endcode

	Scaled copy:
	\code
	tensor_i<2, double> &t1(...), &t2(...);
	tod_copy<2> cp(t1, 0.5);
	cp.perform(t2); // Copies the elements of t1 multiplied by 0.5 to t2
	\endcode

	Permuted copy:
	\code
	tensor_i<2, double> &t1(...), &t2(...);
	permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
	tod_copy<2> cp(t1, perm);
	cp.perform(t2); // Copies transposed t1 to t2
	\endcode

	Permuted and scaled copy:
	\code
	tensor_i<2, double> &t1(...), &t2(...);
	permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
	tod_copy<2> cp(t1, perm, 0.5);
	cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
	\endcode

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_copy
	: public tod_additive<N>, public timings<tod_copy<N> >
{
	friend class timings<tod_copy<N> >;
public:
	static const char *k_clazz; //!< Class name

private:
	struct registers {
		const double *m_ptra;
		double *m_ptrb;
#ifdef LIBTENSOR_DEBUG
		const double *m_ptra_end;
		double *m_ptrb_end;
#endif // LIBTENSOR_DEBUG
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
		loop_list_node()
			: m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb)
			: m_weight(weight), m_inca(inca), m_incb(incb),
			m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb)
			: m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_dcopy : public processor_op_i_t, public timings<op_dcopy> {
	private:
		friend class timings<op_dcopy>;
		static const char* k_clazz;
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_dcopy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_daxpy : public processor_op_i_t, public timings<op_daxpy> {
	private:
		friend class timings<op_daxpy>;
		static const char* k_clazz;
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_daxpy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

private:
	tensor_i<N, double> &m_t; //!< Source %tensor
	tensor_ctrl<N, double> m_tctrl; //!< Source %tensor control
	double m_c; //!< Scaling coefficient
	permutation<N> m_perm; //!< Permutation of elements

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Prepares the copy operation
		\param t Source %tensor.
		\param c Coefficient.
	 **/
	tod_copy(tensor_i<N, double> &t, double c = 1.0);

	/**	\brief Prepares the permute & copy operation
		\param t Source %tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	 **/
	tod_copy(tensor_i<N, double> &t, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_copy();

	//@}

	//!	\name Implementation of
	//!		libtensor::direct_tensor_operation<N, double>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of libtensor::tod_additive<N>
	//@{
	virtual void perform(tensor_i<N, double> &t) throw(exception);
	virtual void perform(tensor_i<N, double> &t, double c) throw(exception);
	//@}

private:
	template<typename CoreOp>
	void do_perform(tensor_i<N, double> &t, double c) throw(exception);
};

template<size_t N>
const char *tod_copy<N>::k_clazz = "tod_copy<N>";
template<size_t N>
const char *tod_copy<N>::op_dcopy::k_clazz = "tod_copy<N>::op_dcopy";
template<size_t N>
const char *tod_copy<N>::op_daxpy::k_clazz = "tod_copy<N>::op_daxpy";

template<size_t N>
inline tod_copy<N>::tod_copy(tensor_i<N, double> &t, double c)
	: m_t(t), m_tctrl(t), m_c(c) {
}

template<size_t N>
inline tod_copy<N>::tod_copy(tensor_i<N, double> &t,
	const permutation<N> &perm, double c)
	: m_t(t), m_tctrl(t), m_c(c), m_perm(perm) {
}

template<size_t N>
tod_copy<N>::~tod_copy() {
}

template<size_t N>
inline void tod_copy<N>::prefetch() throw(exception) {
	m_tctrl.req_prefetch();
}

template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tdst) throw(exception) {
	do_perform<op_dcopy>(tdst, 1.0);
}

template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tdst, double c)
	throw(exception) {
	do_perform<op_daxpy>(tdst, c);
}

template<size_t N> template<typename CoreOp>
void tod_copy<N>::do_perform(tensor_i<N, double> &tdst, double c)
	throw(exception) {

	static const char *method = "do_perform(tensor_i<N, double>&, double)";
	tod_copy<N>::start_timer();

	dimensions<N> dims(m_t.get_dims()); dims.permute(m_perm);
	if(dims != tdst.get_dims()) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incorrect dimensions of the output tensor.");
	}

	tensor_ctrl<N, double> tctrl_dst(tdst);
	const double *psrc = m_tctrl.req_const_dataptr();
	double *pdst = tctrl_dst.req_dataptr();

	permutation<N> inv_perm(m_perm);
	inv_perm.invert();
	size_t ib[N];
	for(size_t i = 0; i < N; i++) ib[i] = i;
	inv_perm.apply(ib);

	loop_list_t lst;
	for(size_t i = 0; i < N; i++) {
		size_t inca = m_t.get_dims().get_increment(i);
		size_t incb = tdst.get_dims().get_increment(ib[i]);
		loop_list_node node(m_t.get_dims()[i], inca, incb);
		if(i < N-1) {
			node.m_op = new op_loop(m_t.get_dims()[i], inca, incb);
		} else {
			node.m_op = new CoreOp(
				m_t.get_dims()[i], inca, incb, c*m_c);
		}
		lst.push_back(node);
	}

	registers regs;
	regs.m_ptra = psrc;
	regs.m_ptrb = pdst;
#ifdef LIBTENSOR_DEBUG
	regs.m_ptra_end = psrc + dims.get_size();
	regs.m_ptrb_end = pdst + dims.get_size();
#endif // LIBTENSOR_DEBUG

	try {
		processor_t proc(lst, regs);
		proc.process_next();
	} catch(exception &e) {
		for(typename loop_list_t::iterator i = lst.begin();
			i != lst.end(); i++) {

			delete i->m_op;
			i->m_op = NULL;
		}
		throw;
	}

	for(typename loop_list_t::iterator i = lst.begin();
		i != lst.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}

	m_tctrl.ret_dataptr(psrc);
	tctrl_dst.ret_dataptr(pdst);

	tod_copy<N>::stop_timer();
}

template<size_t N>
void tod_copy<N>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {

	static const char *clazz = "tod_copy<N>::op_loop";
	static const char *method = "exec(processor_t&, registers&)";

	const double *ptra = regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
#ifdef LIBTENSOR_DEBUG
		if(ptra > regs.m_ptra_end) {
			throw overflow("libtensor", clazz, method, __FILE__,
				__LINE__, "Source buffer overflow.");
		}
		if(ptrb > regs.m_ptrb_end) {
			throw overflow("libtensor", clazz, method, __FILE__,
				__LINE__, "Destination buffer overflow.");
		}
#endif // LIBTENSOR_DEBUG
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}
}

template<size_t N>
void tod_copy<N>::op_dcopy::exec(processor_t &proc, registers &regs)
	throw(exception) {

	tod_copy<N>::op_dcopy::start_timer();
//	static const char *clazz = "tod_copy<N>::op_dcopy";
	static const char *method = "exec(processor_t&, registers&)";

	if(m_len == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(regs.m_ptra + (m_len - 1)*m_inca >= regs.m_ptra_end) {
		throw overflow("libtensor", k_clazz, method, __FILE__, __LINE__,
			"Source buffer overflow.");
	}
	if(regs.m_ptrb + (m_len - 1)*m_incb >= regs.m_ptrb_end) {
		throw overflow("libtensor", k_clazz, method, __FILE__, __LINE__,
			"Destination buffer overflow.");
	}
#endif // LIBTENSOR_DEBUG

	cblas_dcopy(m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
	if(m_c != 1.0) {
		cblas_dscal(m_len, m_c, regs.m_ptrb, m_incb);
	}
	tod_copy<N>::op_dcopy::stop_timer();
}

template<size_t N>
void tod_copy<N>::op_daxpy::exec(processor_t &proc, registers &regs)
	throw(exception) {

	tod_copy<N>::op_daxpy::start_timer();
//	static const char *clazz = "tod_copy<N>::op_daxpy";
	static const char *method = "exec(processor_t&, registers&)";

	if(m_len == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(regs.m_ptra + (m_len - 1)*m_inca >= regs.m_ptra_end) {
		throw overflow("libtensor", k_clazz, method, __FILE__, __LINE__,
			"Source buffer overflow.");
	}
	if(regs.m_ptrb + (m_len - 1)*m_incb >= regs.m_ptrb_end) {
		throw overflow("libtensor", k_clazz, method, __FILE__, __LINE__,
			"Destination buffer overflow.");
	}
#endif // LIBTENSOR_DEBUG

	cblas_daxpy(m_len, m_c, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
	tod_copy<N>::op_daxpy::stop_timer();
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_H

