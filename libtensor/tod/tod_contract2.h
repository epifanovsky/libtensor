#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../linalg.h"
#include "tod_additive.h"
#include "contraction2.h"
#include "contraction2_list_builder.h"
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

	class op_loop_mul :
		public processor_op_i_t, public timings<op_loop_mul> {
	public:
		static const char *k_clazz;
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
	class op_ddot :
		public processor_op_i_t, public timings<op_ddot> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_n;
	public:
		op_ddot(double d, size_t n) : m_d(d), m_n(n) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_i b
	class op_daxpy_a :
		public processor_op_i_t, public timings<op_daxpy_a> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_n, m_stepc;
	public:
		op_daxpy_a(double d, size_t n, size_t stepc) :
			m_d(d), m_n(n), m_stepc(stepc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a b_i
	class op_daxpy_b :
		public processor_op_i_t, public timings<op_daxpy_b> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_n, m_stepc;
	public:
		op_daxpy_b(double d, size_t n, size_t stepc) :
			m_d(d), m_n(n), m_stepc(stepc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_ip b_p
	class op_dgemv_n_a :
		public processor_op_i_t, public timings<op_dgemv_n_a> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda;
	public:
		op_dgemv_n_a(double d, size_t rows, size_t cols,
			size_t stepb, size_t lda) :
			m_d(d), m_rows(rows), m_cols(cols),
			m_stepb(stepb), m_lda(lda) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_pi b_p
	class op_dgemv_t_a :
		public processor_op_i_t, public timings<op_dgemv_t_a> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda, m_stepc;
	public:
		op_dgemv_t_a(double d, size_t rows, size_t cols,
			size_t stepb, size_t lda, size_t stepc) :
			m_d(d), m_rows(rows), m_cols(cols),
			m_stepb(stepb), m_lda(lda), m_stepc(stepc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_p b_ip
	class op_dgemv_n_b :
		public processor_op_i_t, public timings<op_dgemv_n_b> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb;
	public:
		op_dgemv_n_b(double d, size_t rows, size_t cols,
			size_t stepa, size_t ldb) :
			m_d(d), m_rows(rows), m_cols(cols),
			m_stepa(stepa), m_ldb(ldb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_i = a_p b_pi
	class op_dgemv_t_b :
		public processor_op_i_t, public timings<op_dgemv_t_b> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb, m_stepc;
	public:
		op_dgemv_t_b(double d, size_t rows, size_t cols,
			size_t stepa, size_t ldb, size_t stepc) :
			m_d(d), m_rows(rows), m_cols(cols),
			m_stepa(stepa), m_ldb(ldb), m_stepc(stepc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_ij = a_pi b_pj
	class op_dgemm_tn_ab :
		public processor_op_i_t, public timings<op_dgemm_tn_ab> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	public:
		op_dgemm_tn_ab(double d, size_t rowsa, size_t colsb,
			size_t colsa, size_t lda, size_t ldb, size_t ldc) :
			m_d(d), m_rowsa(rowsa), m_colsb(colsb), m_colsa(colsa),
			m_lda(lda), m_ldb(ldb), m_ldc(ldc) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	c_ij = a_pj b_pi
	class op_dgemm_tn_ba :
		public processor_op_i_t, public timings<op_dgemm_tn_ba> {
	public:
		static const char *k_clazz;
	private:
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	public:
		op_dgemm_tn_ba(double d, size_t rowsb, size_t colsa,
			size_t colsb, size_t ldb, size_t lda, size_t ldc) :
			m_d(d), m_rowsb(rowsb), m_colsa(colsa), m_colsb(colsb),
			m_ldb(ldb), m_lda(lda), m_ldc(ldc) { }
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
	void match_l1(double d);
	void match_ddot_l2(double d, size_t w1);
	void match_daxpy_a_l2(double d, size_t w1, size_t k1);
	void match_daxpy_b_l2(double d, size_t w1, size_t k1);
	void match_dgemv_t_a_l3(double d, size_t w1, size_t w2,
		size_t k2w1, size_t k3);
	void match_dgemv_t_b_l3(double d, size_t w1, size_t w2,
		size_t k1, size_t k2w1);
	void match_loops();
	void clean_list();
};


template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::k_clazz = "tod_contract2<N, M, K>";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_loop_mul::k_clazz =
	"tod_contract2<N, M, K>::op_loop_mul";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_ddot::k_clazz =
	"tod_contract2<N, M, K>::op_ddot";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_daxpy_a::k_clazz =
	"tod_contract2<N, M, K>::op_daxpy_a";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_daxpy_b::k_clazz =
	"tod_contract2<N, M, K>::op_daxpy_b";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemv_n_a::k_clazz =
	"tod_contract2<N, M, K>::op_dgemv_n_a";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemv_t_a::k_clazz =
	"tod_contract2<N, M, K>::op_dgemv_t_a";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemv_n_b::k_clazz =
	"tod_contract2<N, M, K>::op_dgemv_n_b";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemv_t_b::k_clazz =
	"tod_contract2<N, M, K>::op_dgemv_t_b";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemm_tn_ab::k_clazz =
	"tod_contract2<N, M, K>::op_dgemm_tn_ab";

template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::op_dgemm_tn_ba::k_clazz =
	"tod_contract2<N, M, K>::op_dgemm_tn_ba";


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
void tod_contract2<N, M, K>::perform(
	tensor_i<k_orderc, double> &tc) throw(exception) {

	do_perform(tc, true, 1.0);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(
	tensor_i<k_orderc, double> &tc, double d) throw(exception) {

	do_perform(tc, false, d);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::do_perform(
	tensor_i<k_orderc, double> &tc, bool zero, double d) throw(exception) {

	tod_contract2<N, M, K>::start_timer();

	loop_list_adapter list_adapter(m_list);
	contraction2_list_builder<N, M, K, loop_list_adapter> lstbld(m_contr);
	lstbld.populate(list_adapter, m_ta.get_dims(), m_tb.get_dims(),
		tc.get_dims());

	tensor_ctrl<k_ordera, double> ctrla(m_ta);
	tensor_ctrl<k_orderb, double> ctrlb(m_tb);
	tensor_ctrl<k_orderc, double> ctrlc(tc);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	if(zero) {
		size_t szc = tc.get_dims().get_size();
		for(size_t i = 0; i < szc; i++) ptrc[i] = 0.0;
	}

	//~ std::cout << "[";
	match_l1(d);
	//~ std::cout << "]" << std::endl;
	match_loops();
	try {
		registers regs;
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		regs.m_ptrc = ptrc;

		processor_t(m_list, regs).process_next();
	} catch(exception e) {
		clean_list();
		tod_contract2<N, M, K>::stop_timer();
		throw;
	}

	clean_list();

	ctrla.ret_dataptr(ptra);
	ctrlb.ret_dataptr(ptrb);
	ctrlc.ret_dataptr(ptrc);

	tod_contract2<N, M, K>::stop_timer();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_l1(double d) {

	typedef typename loop_list_t::iterator list_iter;

	//	1. Find:
	//	-----------
	//	w   a  b  c
	//	w1  1  1  0  -->  c = a_p b_p
	//	-----------       sz(p) = w1
	//	                  [ddot]
	//
	//	2. Minimize k1a:
	//	-------------
	//	w   a  b  c
	//	w1  1  0  k1a  -->  c_i# = a_i b
	//	-------------       sz(i) = w1, sz(#) = k1a
	//	                    [daxpy_a]
	//
	//	3. Minimize k1b:
	//	-------------
	//	w   a  b  c
	//	w1  0  1  k1b  -->  c_i# = a b_i
	//	-------------       sz(i) = w1, sz(#) = k1b
	//	                    [daxpy_b]
	//
	list_iter i1 = m_list.end(), i2 = m_list.end(), i3 = m_list.end();
	size_t k1a_min = 0, k1b_min = 0;
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca == 1 && i->m_incb == 1 && i->m_incc == 0) {
			i1 = i;
		}
		if(i->m_inca == 1 && i->m_incb == 0) {
			if(k1a_min == 0 || k1a_min > i->m_incc) {
				i2 = i; k1a_min = i->m_incc;
			}
		}
		if(i->m_inca == 0 && i->m_incb == 1) {
			if(k1b_min == 0 || k1b_min > i->m_incc) {
				i3 = i; k1b_min = i->m_incc;
			}
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << "ddot";
		i1->m_op = new op_ddot(d, i1->m_weight);
		match_ddot_l2(d, i1->m_weight);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
	if(i2 != m_list.end() && k1b_min != 1) {
		//~ std::cout << "daxpy_a";
		i2->m_op = new op_daxpy_a(d, i2->m_weight, i2->m_incc);
		match_daxpy_a_l2(d, i2->m_weight, i2->m_incc);
		m_list.splice(m_list.end(), m_list, i2);
		return;
	}
	if(i3 != m_list.end()) {
		//~ std::cout << "daxpy_b";
		i3->m_op = new op_daxpy_b(d, i3->m_weight, i3->m_incc);
		match_daxpy_b_l2(d, i3->m_weight, i3->m_incc);
		m_list.splice(m_list.end(), m_list, i3);
		return;
	}

	//	Actually, we should not make it here. Throw an exception.

	//	4. Find
	//	------------
	//	w   a  b   c
	//	w1  1  k1  0  -->  loop_mul: c = a_p b_p#, sz(#) = k1
	//	------------
	//	w   a  b  c
	//	w1  1  0  k1  -->  loop_mul: c_i# = a_i b, sz(#) = k1
	//	------------
	list_iter i4 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca == 1) {
			i4 = i; break;
		}
	}
	//~ std::cout << "loop_mul";
	i4->m_op = new op_loop_mul(d, i4->m_weight,
		i4->m_inca, i4->m_incb, i4->m_incc);
	m_list.splice(m_list.end(), m_list, i4);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_ddot_l2(double d, size_t w1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	-----------
	//	w   a  b  c
	//	w1  1  1  0  -->  c = a_p b_p
	//	-----------       sz(p) = w1
	//	                  [ddot]
	//

	//	1. Minimize k1a:
	//	----------------
	//	w   a       b  c
	//	w1  1       1  0
	//	w2  k1a*w1  0  1  -->  c_i = a_i$p b_p
	//	----------------       sz(i) = w2, sz(p) = w1, sz(%p) = k1a
	//	                       [dgemv_n_a]
	//
	//	2. Minimize k1b:
	//	----------------
	//	w   a  b       c
	//	w1  1  1       0
	//	w2  0  k1b*w1  1  -->  c_i = a_p b_i%p
	//	----------------       sz(i) = w2, sz(p) = w1, sz(%p) = k1b
	//	                       [dgemv_n_b]
	//	                    
	size_t k1a_min = 0, k1b_min = 0;
	list_iter i1 = m_list.end(), i2 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incc != 1) continue;
		if(i->m_incb == 0 && i->m_inca % w1 == 0) {
			register size_t k1a = i->m_inca / w1;
			if(k1a_min == 0 || k1a_min > k1a) {
				k1a_min = k1a; i1 = i;
			}
		}
		if(i->m_inca == 0 && i->m_incb % w1 == 0) {
			register size_t k1b = i->m_incb / w1;
			if(k1b_min == 0 || k1b_min > k1b) {
				k1b_min = k1b; i2 = i;
			}
		}
	}

	if(i1 != m_list.end()) {
		//~ std::cout << " dgemv_n_a";
		i1->m_op = new op_dgemv_n_a(
			d, i1->m_weight, w1, 1, i1->m_inca);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}

	if(i2 != m_list.end()) {
		//~ std::cout << " dgemv_n_b";
		i2->m_op = new op_dgemv_n_b(
			d, i2->m_weight, w1, 1, i2->m_incb);
		m_list.splice(m_list.end(), m_list, i2);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_daxpy_a_l2(
	double d, size_t w1, size_t k1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	w1  1  0  k1  -->  c_i# = a_i b
	//	------------       sz(i) = w1, sz(#) = k1
	//	                   [daxpy_a]
	//

	//	1. Minimize k2:
	//	-----------------
	//	w   a      b   c
	//	w1  1      0   k1
	//	w2  k2*w1  k3  0   -->  c_i# = a_p$i b_p%
	//	-----------------       sz(i) = w1, sz(p) = w2,
	//	                        sz(#) = k1, sz($) = k2, sz(%) = k3
	//	                        [dgemv_t_a]
	//
	size_t k2min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incc != 0) continue;
		if(i->m_inca % w1 != 0) continue;

		register size_t k2 = i->m_inca / w1;
		if(k2min == 0 || k2min > k2) {
			k2min = k2; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemv_t_a";
		i1->m_op = new op_dgemv_t_a(d, i1->m_weight, w1,
			i1->m_incb, i1->m_inca, k1);
		//~ if(k1 == 1) match_dgemv_t_a_l3(
			//~ d, w1, i1->m_weight, i1->m_inca, i1->m_incb);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_daxpy_b_l2(
	double d, size_t w1, size_t k1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	w1  0  1  k1  -->  c_i# = a b_i
	//	------------       sz(i) = w1, sz(#) = k1
	//	                   [daxpy_b]

	//	1. Minimize k2:
	//	-----------------
	//	w   a   b      c
	//	w1  0   1      k1
	//	w2  k3  k2*w1  0  -->  c_i# = a_p$ b_p%i
	//      -----------------      sz(i) = w1, sz(p) = w2
	//	                       sz(#) = k1, sz($) = k3, sz(%) = k2
	//	                       [dgemv_t_b]
	size_t k2min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incc != 0) continue;
		if(i->m_incb % w1 != 0) continue;

		register size_t k2 = i->m_incb / w1;
		if(k2min == 0 || k2min > k2) {
			k2min = k2; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemv_t_b";
		i1->m_op = new op_dgemv_t_b(d, i1->m_weight, w1,
			i1->m_inca, i1->m_incb, k1);
		//~ match_dgemv_t_b_l3(
			//~ d, w1, i1->m_weight, i1->m_inca, i1->m_incb);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_t_a_l3(
	double d, size_t w1, size_t w2, size_t k2w1, size_t k3) {

	typedef typename loop_list_t::iterator list_iter;

	//	Invoked when the pattern is found:
	//	---------------
	//	w   a     b   c
	//	w1  1     0   1
	//	w2  k2w1  k3  0  -->  c_i = a_p$i b_p%
	//	---------------       sz(i) = w1, sz(p) = w2,
	//	                      sz($) = k2, sz(%) = k3
	//	                      [dgemv_t_a]
	//

	//	1. Minimize k4 in:
	//	------------------------
	//	w   a      b       c
	//	w1  1      0       1
	//	w2  k2*w1  k3'*w3  0
	//	w3  0      1       k4*w1  --> c_j#i = a_p$i b_p%j
	//	------------------------      sz(i) = w1, sz(j) = w3,
	//	                              sz(p) = w2,
	//	                              sz(#) = k4, sz($) = k2,
	//	                              sz(%) = k3'
	//	                              [dgemm_tn_ba]
	//
	size_t k4min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 0 || i->m_incb != 1) continue;
		if(k3 % i->m_weight != 0) continue;
		if(i->m_incc % w1 != 0) continue;

		register size_t k4 = i->m_incc / w1;
		if(k4min == 0 || k4min > k4) {
			k4min = k4; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_tn_ba";
		i1->m_op = new op_dgemm_tn_ba(
			d, w2, w1, i1->m_weight, k3, k2w1, i1->m_incc);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_t_b_l3(
	double d, size_t w1, size_t w2, size_t k1, size_t k2w1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Invoked when a pattern is found:
	//	----------------
	//	w   a   b      c
	//	w1  0   1      1  -->  daxpy_b: c_i = a b_i, sz(i) = w1
	//	w2  k1  k2w1   0  -->  dgemv_t_b: c_i = a_p# b_p$i
	//	                                  sz(#) = k1
	//	                                  sz($j) = k2w1
	//	----------------
	//

	//	1. Find with the minimum k3
	//	----------------------
	//	w   a      b     c
	//	w1  0      1     1
	//	w2  k4*w3  k2w1  0
	//	w3  1      0     k3*w1  -->  dgemm_tn: c_i%j = a_p#i b_p$j
	//	                                        sz(#i) = k4*w3
	//	                                        sz($j) = k2w1
	//	                                        sz(%j) = k3*w1
	//	----------------------
	//
	size_t k3min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 1 || i->m_incb != 0) continue;
		if(k1 % i->m_weight != 0) continue;
		if(i->m_incc % w1 != 0) continue;

		register size_t k3 = i->m_incc / w1;
		if(k3min == 0 || k3min > k3) {
			k3min = k3; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_tn_ab";
		i1->m_op = new op_dgemm_tn_ab(
			d, i1->m_weight, w1, w2, k1, k2w1, i1->m_incc);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
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

	tod_contract2<N, M, K>::op_loop_mul::start_timer();
	const double *ptra = regs.m_ptra, *ptrb = regs.m_ptrb;
	double *ptrc = regs.m_ptrc;

	for(size_t i = 0; i < m_len; i++) {
		ptrc[0] += m_d * ptra[0] * ptrb[0];
		ptra += m_inca;
		ptrb += m_incb;
		ptrc += m_incc;
	}

	regs.m_ptra = ptra;
	regs.m_ptrb = ptrb;
	regs.m_ptrc = ptrc;
	tod_contract2<N, M, K>::op_loop_mul::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_ddot::exec(
	processor_t &proc, registers &regs) throw(exception) {

	op_ddot::start_timer();
	regs.m_ptrc[0] +=
		m_d * cblas_ddot(m_n, regs.m_ptra, 1, regs.m_ptrb, 1);
	op_ddot::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_daxpy_a::exec(
	processor_t &proc, registers &regs) throw(exception) {

	op_daxpy_a::start_timer();
	cblas_daxpy(m_n, regs.m_ptrb[0]*m_d, regs.m_ptra, 1,
		regs.m_ptrc, m_stepc);
	op_daxpy_a::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_daxpy_b::exec(
	processor_t &proc, registers &regs) throw(exception) {

	op_daxpy_b::start_timer();
	cblas_daxpy(m_n, regs.m_ptra[0]*m_d, regs.m_ptrb, 1,
		regs.m_ptrc, m_stepc);
	op_daxpy_b::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_n_a::exec(
	processor_t &proc, registers &regs) throw(exception) {

	op_dgemv_n_a::start_timer();
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_rows, m_cols, m_d,
		regs.m_ptra, m_lda, regs.m_ptrb, m_stepb, 1.0, regs.m_ptrc, 1);
	op_dgemv_n_a::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_t_a::exec(
	processor_t &proc, registers &regs) throw(exception) {

	op_dgemv_t_a::start_timer();
	cblas_dgemv(CblasRowMajor, CblasTrans, m_rows, m_cols, m_d,
		regs.m_ptra, m_lda, regs.m_ptrb, m_stepb, 1.0,
		regs.m_ptrc, m_stepc);
	op_dgemv_t_a::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_n_b::exec(
	processor_t &proc, registers &regs) throw(exception) {

	timings<op_dgemv_n_b>::start_timer();
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m_rows, m_cols, m_d,
		regs.m_ptrb, m_ldb, regs.m_ptra, m_stepa, 1.0, regs.m_ptrc, 1);
	timings<op_dgemv_n_b>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemv_t_b::exec(
	processor_t &proc, registers &regs) throw(exception) {

	timings<op_dgemv_t_b>::start_timer();
	cblas_dgemv(CblasRowMajor, CblasTrans, m_rows, m_cols, m_d,
		regs.m_ptrb, m_ldb, regs.m_ptra, m_stepa, 1.0,
		regs.m_ptrc, m_stepc);
	timings<op_dgemv_t_b>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemm_tn_ab::exec(
	processor_t &proc, registers &regs) throw(exception) {

	timings<op_dgemm_tn_ab>::start_timer();
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		m_rowsa, m_colsb, m_colsa, m_d,
		regs.m_ptra, m_lda, regs.m_ptrb, m_ldb,
		1.0, regs.m_ptrc, m_ldc);
	timings<op_dgemm_tn_ab>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::op_dgemm_tn_ba::exec(
	processor_t &proc, registers &regs) throw(exception) {

	timings<op_dgemm_tn_ba>::start_timer();
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		m_rowsb, m_colsa, m_colsb, m_d,
		regs.m_ptrb, m_ldb, regs.m_ptra, m_lda,
		1.0, regs.m_ptrc, m_ldc);
	timings<op_dgemm_tn_ba>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

