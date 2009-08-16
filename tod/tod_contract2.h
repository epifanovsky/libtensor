#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "tod_additive.h"
#include "contraction2.h"
#include "processor.h"

namespace libtensor {

/**	\brief Contracts two tensors (double)

	\tparam N Order of the first %tensor (a) less the contraction degree
	\tparam M Order of the second %tensor (b) less the contraction degree
	\tparam K Contraction degree (the number of indexes over which the
		tensors are contracted)

	This operation contracts %tensor T1 permuted as P1 with %tensor T2
	permuted as P2 over n last indexes. The result is permuted as Pres
	and written or added to the resulting %tensor.

	Although it is convenient to define a contraction through permutations,
	it is not the most efficient way of calculating it. This class seeks
	to use algorithms tailored for different tensors to get the best
	performance. For more information, read the wiki section on %tensor
	contractions.

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_contract2
	: public tod_additive<N+M>,
		public timings<tod_contract2<N,M,K> >
{
private:
	friend class timings<tod_contract2<N,M,K> >;
	static const char* k_clazz;

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
		size_t m_inca, m_incb, m_incc;
		processor_op_i_t *m_op;
		loop_list_node() : m_weight(0), m_inca(0), m_incb(0),
			m_incc(0), m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb,
			size_t incc) : m_weight(weight), m_inca(inca),
			m_incb(incb), m_incc(incc), m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop
		: public processor_op_i_t  {
	private:
		size_t m_len, m_inca, m_incb, m_incc;
	public:
		op_loop(size_t len, size_t inca, size_t incb, size_t incc) :
			m_len(len), m_inca(inca), m_incb(incb), m_incc(incc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_loop_mul
		: public processor_op_i_t {
	private:
		double m_d;
		size_t m_len, m_inca, m_incb, m_incc;
	public:
		op_loop_mul(double d, size_t len, size_t inca, size_t incb,
			size_t incc) : m_d(d), m_len(len), m_inca(inca),
			m_incb(incb), m_incc(incc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c = a_i b_i
	class op_ddot : public processor_op_i_t, public timings<op_ddot> {
	private:
		friend class timings<op_ddot>;
		static const char* k_clazz;
		double m_d;
		size_t m_n;
	public:
		op_ddot(double d, size_t n) : m_d(d), m_n(n) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_i b
	class op_daxpy_a : public processor_op_i_t, public timings<op_daxpy_a> {
	private:
		friend class timings<op_daxpy_a>;
		static const char* k_clazz;
		double m_d;
		size_t m_n;
	public:
		op_daxpy_a(double d, size_t n) : m_d(d), m_n(n) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a b_i
	class op_daxpy_b : public processor_op_i_t, public timings<op_daxpy_b> {
	private:
		friend class timings<op_daxpy_b>;
		static const char* k_clazz;
		double m_d;
		size_t m_n;
	public:
		op_daxpy_b(double d, size_t n) : m_d(d), m_n(n) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_ip b_p
	class op_dgemv_a : public processor_op_i_t, public timings<op_dgemv_a> {
	private:
		friend class timings<op_dgemv_a>;
		static const char* k_clazz;
		double m_d;
		size_t m_rows, m_cols, m_lda;
	public:
		op_dgemv_a(double d, size_t rows, size_t cols, size_t lda) :
			m_d(d), m_rows(rows), m_cols(cols), m_lda(lda) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_p b_ip
	class op_dgemv_b : public processor_op_i_t, public timings<op_dgemv_b> {
	private:
		friend class timings<op_dgemv_b>;
		static const char* k_clazz;
		double m_d;
		size_t m_rows, m_cols, m_ldb;
	public:
		op_dgemv_b(double d, size_t rows, size_t cols, size_t ldb) :
			m_d(d), m_rows(rows), m_cols(cols), m_ldb(ldb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class loop_list_adapter {
	private:
		loop_list_t &m_list;

	public:
		loop_list_adapter(loop_list_t &list) : m_list(list) { }
		void append(size_t weight, size_t inca, size_t incb,
			size_t incc);
	};

public:
	static const size_t k_ordera = N + K;
	static const size_t k_orderb = M + K;
	static const size_t k_orderc = N + M;

private:

	contraction2<N, M, K> m_contr; //!< Contraction
	tensor_i<k_ordera, double> &m_ta; //!< First tensor (a)
	tensor_i<k_orderb, double> &m_tb; //!< Second tensor (b)
	loop_list_t m_list; //!< Loop list

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract2(const contraction2<N, M, K> &contr,
		tensor_i<k_ordera, double> &ta, tensor_i<k_orderb, double> &tb);


	/**	\brief Virtual destructor
	 **/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive<N+M>
	//@{
	virtual void perform(tensor_i<k_orderc, double> &tc) throw(exception);
	virtual void perform(tensor_i<k_orderc, double> &tc, double d)
		throw(exception);
	//@}

private:
	void do_perform(tensor_i<k_orderc, double> &tc, bool zero, double d)
		throw(exception);
	void match_level_1(double d);
	void match_ddot_level_2(double d, size_t w0);
	void match_loops();
	void clean_list();
};

template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::k_clazz="tod_contract2<N,M,K>";
template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::op_ddot::k_clazz="tod_contract2<N,M,K>::op_ddot";
template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::op_daxpy_a::k_clazz="tod_contract2<N,M,K>::op_daxpy_a";
template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::op_daxpy_b::k_clazz="tod_contract2<N,M,K>::op_daxpy_b";
template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::op_dgemv_a::k_clazz="tod_contract2<N,M,K>::op_dgemv_a";
template<size_t N, size_t M, size_t K>
const char* tod_contract2<N,M,K>::op_dgemv_b::k_clazz="tod_contract2<N,M,K>::op_dgemv_b";

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::tod_contract2(const contraction2<N,M,K> &contr,
	tensor_i<k_ordera,double> &ta, tensor_i<k_orderb,double> &tb) :
	m_contr(contr), m_ta(ta), m_tb(tb) {
}

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::~tod_contract2() {
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::prefetch() throw(exception) {
	tensor_ctrl<k_ordera, double> ctrl_ta(m_ta);
	tensor_ctrl<k_orderb, double> ctrl_tb(m_tb);
	ctrl_ta.req_prefetch();
	ctrl_tb.req_prefetch();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(tensor_i<k_orderc, double> &tc)
	throw(exception) {

	do_perform(tc, true, 1.0);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(tensor_i<k_orderc, double> &tc, double d)
	throw(exception) {

	do_perform(tc, false, d);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::do_perform(tensor_i<k_orderc, double> &tc,
	bool zero, double d) throw(exception) {

	tod_contract2<N,M,K>::start_timer();

	loop_list_adapter list_adapter(m_list);
	m_contr.populate(list_adapter, m_ta.get_dims(), m_tb.get_dims(),
		tc.get_dims());

	tensor_ctrl<N+K, double> ctrla(m_ta);
	tensor_ctrl<M+K, double> ctrlb(m_tb);
	tensor_ctrl<N+M, double> ctrlc(tc);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	if(zero) {
		size_t szc = tc.get_dims().get_size();
		for(size_t i=0; i<szc; i++) ptrc[i] = 0.0;
	}

	match_level_1(d);
	match_loops();
	try {
		registers regs;
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		regs.m_ptrc = ptrc;

		processor_t(m_list, regs).process_next();
	} catch(exception e) {
		clean_list();
		throw;
	}

	clean_list();

	ctrla.ret_dataptr(ptra);
	ctrlb.ret_dataptr(ptrb);
	ctrlc.ret_dataptr(ptrc);

	tod_contract2<N,M,K>::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_level_1(double d) {
	bool lastc_found = false;
	typename loop_list_t::iterator lasta, lastb, lastc;

	for(typename loop_list_t::iterator i = m_list.begin();
		i != m_list.end(); i++) {

		if(i->m_inca == 1) lasta = i;
		if(i->m_incb == 1) lastb = i;
		if(i->m_incc == 1) { lastc = i; lastc_found = true; }
	}

	if(lasta == lastb) {
		lasta->m_op = new op_ddot(d, lasta->m_weight);
		match_ddot_level_2(d, lasta->m_weight);
		m_list.splice(m_list.end(), m_list, lasta);
	} else {
		if(lastc_found && lasta == lastc) {
			lasta->m_op = new op_daxpy_a(d, lasta->m_weight);
			m_list.splice(m_list.end(), m_list, lasta);
		} else
		if(lastc_found && lastb == lastc) {
			lastb->m_op = new op_daxpy_b(d, lastb->m_weight);
			m_list.splice(m_list.end(), m_list, lastb);
		} else {
			lasta->m_op = new op_loop_mul(d, lasta->m_weight,
				lasta->m_inca, lasta->m_incb, lasta->m_incc);
			m_list.splice(m_list.end(), m_list, lasta);
		}
	}
}

template<size_t N, size_t M, size_t K>
inline void tod_contract2<N, M, K>::match_ddot_level_2(double d, size_t w0) {
	bool found_match = false;
	for(typename loop_list_t::iterator i = m_list.begin();
		i != m_list.end(); i++) {

		if(i->m_incc == 1) {
			if(i->m_inca != 0) {
				i->m_op = new op_dgemv_a(d, i->m_weight,
					w0, i->m_inca);
				found_match = true;
				m_list.splice(m_list.end(), m_list, i);
				break;
			} else if(i->m_incb != 0) {
				i->m_op = new op_dgemv_b(d, i->m_weight,
					w0, i->m_incb);
				found_match = true;
				m_list.splice(m_list.end(), m_list, i);
				break;
			}
		}
	}
	if(found_match); // match_ddot_level_2 here
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_loops() {
	for(typename loop_list_t::iterator i = m_list.begin();
		i != m_list.end(); i++) {

		if(i->m_op == NULL) {
			i->m_op = new op_loop(i->m_weight, i->m_inca,
				i->m_incb, i->m_incc);
		}
	}
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::clean_list() {
	for(typename loop_list_t::iterator i = m_list.begin();
		i != m_list.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}
}

template<size_t N, size_t M, size_t K>
inline void tod_contract2<N, M, K>::loop_list_adapter::append(size_t weight,
	size_t inca, size_t incb, size_t incc) {
	m_list.push_back(loop_list_node(weight, inca, incb, incc));
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {
	const double *ptra = regs.m_ptra, *ptrb = regs.m_ptrb;
	double *ptrc = regs.m_ptrc;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		regs.m_ptrc = ptrc;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
		ptrc += m_incc;
	}
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_loop_mul::exec(processor_t &proc,
	registers &regs) throw(exception) {
	const double *ptra = regs.m_ptra, *ptrb = regs.m_ptrb;
	double *ptrc = regs.m_ptrc;

	for(size_t i=0; i<m_len; i++) {
		*ptrc += (*ptra)*(*ptrb);
		ptra += m_inca;
		ptrb += m_incb;
		ptrc += m_incc;
	}

	regs.m_ptra = ptra;
	regs.m_ptrb = ptrb;
	regs.m_ptrc = ptrc;
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_ddot::exec(processor_t &proc, registers &regs)
	throw(exception) {
	tod_contract2<N, M, K>::op_ddot::start_timer();
	*(regs.m_ptrc) = m_d*cblas_ddot(m_n, regs.m_ptra, 1, regs.m_ptrb, 1);
	tod_contract2<N, M, K>::op_ddot::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_daxpy_a::exec(processor_t &proc,
	registers &regs) throw(exception) {
	tod_contract2<N, M, K>::op_daxpy_a::start_timer();
	cblas_daxpy(m_n, *(regs.m_ptrb)*m_d, regs.m_ptra, 1, regs.m_ptrc, 1);
	tod_contract2<N, M, K>::op_daxpy_a::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_daxpy_b::exec(processor_t &proc,
	registers &regs) throw(exception) {
	tod_contract2<N, M, K>::op_daxpy_b::start_timer();
	cblas_daxpy(m_n, *(regs.m_ptra)*m_d, regs.m_ptrb, 1, regs.m_ptrc, 1);
	tod_contract2<N, M, K>::op_daxpy_b::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_a::exec(processor_t &proc,
	registers &regs) throw(exception) {
	tod_contract2<N, M, K>::op_dgemv_a::start_timer();
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_rows, m_cols, m_d,
		regs.m_ptra, m_lda, regs.m_ptrb, 1, 1.0, regs.m_ptrc, 1);
	tod_contract2<N, M, K>::op_dgemv_a::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_b::exec(processor_t &proc,
	registers &regs) throw(exception) {
	tod_contract2<N, M, K>::op_dgemv_b::start_timer();
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_rows, m_cols, m_d,
		regs.m_ptrb, m_ldb, regs.m_ptra, 1, 1.0, regs.m_ptrc, 1);
	tod_contract2<N, M, K>::op_dgemv_b::stop_timer();
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

