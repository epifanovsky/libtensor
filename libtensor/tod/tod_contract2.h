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
	typedef typename loop_list_t::iterator loop_list_iterator_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incb, m_incc;
		void (tod_contract2<N, M, K>::*m_fn)(registers &);
		loop_list_node() : m_weight(0), m_inca(0), m_incb(0),
			m_incc(0), m_fn(0) { }
		loop_list_node(size_t weight, size_t inca, size_t incb,
			size_t incc) : m_weight(weight), m_inca(inca),
			m_incb(incb), m_incc(incc), m_fn(0) { }
	};

	//!	c = a_i b_i
	struct {
		double m_d;
		size_t m_n;
	} m_ddot;

	//!	c_i = a_i b
	struct {
		double m_d;
		size_t m_n, m_stepc;
	} m_daxpy_a;

	//!	c_i = a b_i
	struct {
		double m_d;
		size_t m_n, m_stepc;
	} m_daxpy_b;

	//!	c_i = a_ip b_p
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda;
	} m_dgemv_n_a;

	//!	c_i = a_pi b_p
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepb, m_lda, m_stepc;
	} m_dgemv_t_a;

	//!	c_i = a_p b_ip
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb;
	} m_dgemv_n_b;

	//!	c_i = a_p b_pi
	struct {
		double m_d;
		size_t m_rows, m_cols, m_stepa, m_ldb, m_stepc;
	} m_dgemv_t_b;

	//!	c_ij = a_ip b_jp
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_nt_ab;

	//!	c_ij = a_pi b_pj
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_tn_ab;

	//!	c_ij = a_pi b_jp
	struct {
		double m_d;
		size_t m_rowsa, m_colsb, m_colsa, m_lda, m_ldb, m_ldc;
	} m_dgemm_tt_ab;

	//!	c_ij = a_pj b_ip
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_nn_ba;

	//!	c_ij = a_jp b_ip
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_nt_ba;

	//!	c_ij = a_pj b_pi
	struct {
		double m_d;
		size_t m_rowsb, m_colsa, m_colsb, m_ldb, m_lda, m_ldc;
	} m_dgemm_tn_ba;

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
	const char *m_kernelname;

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
	void match_dgemv_n_a_l3(double d, size_t w1, size_t w2, size_t k1w1);
	void match_dgemv_n_b_l3(double d, size_t w1, size_t w2, size_t k1w1);
	void match_dgemv_t_a1_l3(double d, size_t w1, size_t w2,
		size_t k1, size_t k2w1);
	void match_dgemv_t_a2_l3(double d, size_t w1, size_t w2,
		size_t k2w1, size_t k3);
	void match_dgemv_t_b_l3(double d, size_t w1, size_t w2,
		size_t k2w1, size_t k3);

private:
	void exec(loop_list_iterator_t &i, registers &regs);
	void fn_loop(loop_list_iterator_t &i, registers &regs);
	void fn_ddot(registers &r);
	void fn_daxpy_a(registers &r);
	void fn_daxpy_b(registers &r);
	void fn_dgemv_n_a(registers &r);
	void fn_dgemv_t_a(registers &r);
	void fn_dgemv_n_b(registers &r);
	void fn_dgemv_t_b(registers &r);
	void fn_dgemm_nt_ab(registers &r);
	void fn_dgemm_tn_ab(registers &r);
	void fn_dgemm_tt_ab(registers &r);
	void fn_dgemm_nn_ba(registers &r);
	void fn_dgemm_nt_ba(registers &r);
	void fn_dgemm_tn_ba(registers &r);
};


template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::k_clazz = "tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::tod_contract2(const contraction2<N,M,K> &contr,
	tensor_i<k_ordera,double> &ta, tensor_i<k_orderb,double> &tb) :
	m_contr(contr), m_ta(ta), m_tb(tb), m_kernelname(0) {
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
		tod_contract2<N, M, K>::start_timer("zero");
		size_t szc = tc.get_dims().get_size();
		for(size_t i = 0; i < szc; i++) ptrc[i] = 0.0;
		tod_contract2<N, M, K>::stop_timer("zero");
	}

	tod_contract2<N, M, K>::start_timer("match_patterns");
	//~ std::cout << "[";
	m_kernelname = 0;
	match_l1(d);
	if(m_kernelname == 0) m_kernelname = "unknown";
	//~ std::cout << "]" << std::endl;
	tod_contract2<N, M, K>::stop_timer("match_patterns");
	tod_contract2<N, M, K>::start_timer(m_kernelname);
	try {
		registers regs;
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		regs.m_ptrc = ptrc;

		loop_list_iterator_t i = m_list.begin();
		if(i != m_list.end()) {
			exec(i, regs);
		}
	} catch(exception e) {
		tod_contract2<N, M, K>::stop_timer(m_kernelname);
		tod_contract2<N, M, K>::stop_timer();
		throw;
	}
	tod_contract2<N, M, K>::stop_timer(m_kernelname);

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
		m_kernelname = "ddot";
		i1->m_fn = &tod_contract2<N, M, K>::fn_ddot;
		m_ddot.m_d = d;
		m_ddot.m_n = i1->m_weight;
		match_ddot_l2(d, i1->m_weight);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
	if(i2 != m_list.end() && k1b_min != 1) {
		//~ std::cout << "daxpy_a";
		m_kernelname = "daxpy_a";
		i2->m_fn = &tod_contract2<N, M, K>::fn_daxpy_a;
		m_daxpy_a.m_d = d;
		m_daxpy_a.m_n = i2->m_weight;
		m_daxpy_a.m_stepc = i2->m_incc;
		match_daxpy_a_l2(d, i2->m_weight, i2->m_incc);
		m_list.splice(m_list.end(), m_list, i2);
		return;
	}
	if(i3 != m_list.end()) {
		//~ std::cout << "daxpy_b";
		m_kernelname = "daxpy_b";
		i3->m_fn = &tod_contract2<N, M, K>::fn_daxpy_b;
		m_daxpy_b.m_d = d;
		m_daxpy_b.m_n = i3->m_weight;
		m_daxpy_b.m_stepc = i3->m_incc;
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
	//~ list_iter i4 = m_list.end();
	//~ for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		//~ if(i->m_inca == 1) {
			//~ i4 = i; break;
		//~ }
	//~ }
	//~ std::cout << "loop_mul";
	//~ i4->m_op = new op_loop_mul(d, i4->m_weight,
		//~ i4->m_inca, i4->m_incb, i4->m_incc);
	//~ m_list.splice(m_list.end(), m_list, i4);
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
	//	----------------       sz(i) = w2, sz(p) = w1, sz($) = k1a
	//	                       [dgemv_n_a]
	//
	//	2. Minimize k1b:
	//	----------------
	//	w   a  b       c
	//	w1  1  1       0
	//	w2  0  k1b*w1  1  -->  c_i = a_p b_i%p
	//	----------------       sz(i) = w2, sz(p) = w1, sz(%) = k1b
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
		m_kernelname = "dgemv_n_a";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemv_n_a;
		m_dgemv_n_a.m_d = d;
		m_dgemv_n_a.m_rows = i1->m_weight;
		m_dgemv_n_a.m_cols = w1;
		m_dgemv_n_a.m_stepb = 1;
		m_dgemv_n_a.m_lda = i1->m_inca;
		match_dgemv_n_a_l3(d, w1, i1->m_weight, i1->m_inca);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}

	if(i2 != m_list.end()) {
		//~ std::cout << " dgemv_n_b";
		m_kernelname = "dgemv_n_b";
		i2->m_fn = &tod_contract2<N, M, K>::fn_dgemv_n_b;
		m_dgemv_n_b.m_d = d;
		m_dgemv_n_b.m_rows = i2->m_weight;
		m_dgemv_n_b.m_cols = w1;
		m_dgemv_n_b.m_stepa = 1;
		m_dgemv_n_b.m_ldb = i2->m_incb;
		match_dgemv_n_b_l3(d, w1, i2->m_weight, i2->m_incb);
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

	//	1. Minimize k2a:
	//	------------------
	//	w   a       b   c
	//	w1  1       0   k1
	//	w2  k2a*w1  1   0   -->  c_i# = a_p$i b_p
	//	------------------       sz(i) = w1, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2a
	//	                         [dgemv_t_a]
	//
	//	2. Minimize k2b:
	//	------------------
	//	w   a       b   c
	//	w1  1       0   k1
	//	w2  k2b*w1  k3  0   -->  c_i# = a_p$i b_p%
	//	------------------       sz(i) = w1, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2b, sz(%) = k3
	//	                         [dgemv_t_a]
	//	-----------------
	//	w   a       b   c
	//	w1  1       0   1
	//	w2  k2b*w1  k3  0  -->  c_i = a_p$i b_p%
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz($) = k2b, sz(%) = k3
	//	                        [dgemv_t_a]
	//
	size_t k2a_min = 0, k2b_min = 0;
	list_iter i1 = m_list.end(), i2 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incc != 0) continue;
		if(i->m_inca % w1 != 0) continue;

		register size_t k2 = i->m_inca / w1;
		if(i->m_incb == 1 && (k2a_min == 0 || k2a_min > k2)) {
			k2a_min = k2; i1 = i;
		}
		if(k2b_min == 0 || k2b_min > k2) {
			k2b_min = k2; i2 = i;
		}
	}
	if(i1 != m_list.end() && !(k1 == 1 && i2 != m_list.end())) {
		//~ std::cout << " dgemv_t_a1";
		m_kernelname = "dgemv_t_a";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemv_t_a;
		m_dgemv_t_a.m_d = d;
		m_dgemv_t_a.m_rows = i1->m_weight;
		m_dgemv_t_a.m_cols = w1;
		m_dgemv_t_a.m_stepb = 1;
		m_dgemv_t_a.m_lda = i1->m_inca;
		m_dgemv_t_a.m_stepc = k1;
		match_dgemv_t_a1_l3(d, w1, i1->m_weight, k1, i1->m_inca);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
	if(i2 != m_list.end()) {
		//~ std::cout << " dgemv_t_a2";
		m_kernelname = "dgemv_t_a";
		i2->m_fn = &tod_contract2<N, M, K>::fn_dgemv_t_a;
		m_dgemv_t_a.m_d = d;
		m_dgemv_t_a.m_rows = i2->m_weight;
		m_dgemv_t_a.m_cols = w1;
		m_dgemv_t_a.m_stepb = i2->m_incb;
		m_dgemv_t_a.m_lda = i2->m_inca;
		m_dgemv_t_a.m_stepc = k1;
		if(k1 == 1) match_dgemv_t_a2_l3(
			d, w1, i2->m_weight, i2->m_inca, i2->m_incb);
		m_list.splice(m_list.end(), m_list, i2);
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
	size_t k2_min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incc != 0) continue;
		if(i->m_incb % w1 != 0) continue;

		register size_t k2 = i->m_incb / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemv_t_b";
		m_kernelname = "dgemv_t_b";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemv_t_b;
		m_dgemv_t_b.m_d = d;
		m_dgemv_t_b.m_rows = i1->m_weight;
		m_dgemv_t_b.m_cols = w1;
		m_dgemv_t_b.m_stepa = i1->m_inca;
		m_dgemv_t_b.m_ldb = i1->m_incb;
		m_dgemv_t_b.m_stepc = k1;
		if(k1 == 1) match_dgemv_t_b_l3(
			d, w1, i1->m_weight, i1->m_incb, i1->m_inca);
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_n_a_l3(
	double d, size_t w1, size_t w2, size_t k1w1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	---------------
	//	w   a      b  c
	//	w1  1      1  0
	//	w2  k1*w1  0  1  -->  c_i = a_i$p b_p
	//	---------------       sz(i) = w2, sz(p) = w1, sz($) = k1
	//	                      [dgemv_n_a]

	//	1. Minimize k2:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      1      0
	//	w2  k1*w1  0      1
	//	w3  0      k2*w1  k3*w2  -->  c_j#i = a_i$p b_j%p
	//	-----------------------       sz(i) = w2, sz(j) = w3,
	//	                              sz(p) = w1
	//	                              sz(#) = k3, sz($) = k1,
	//	                              sz(%) = k2
	//	                              [dgemm_nt_ba]
	//
	size_t k2_min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 0) continue;
		if(i->m_incc % w2 != 0) continue;
		if(i->m_incb % w1 != 0) continue;

		register size_t k2 = i->m_incb / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_nt_ba";
		m_kernelname = "dgemm_nt_ba";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemm_nt_ba;
		m_dgemm_nt_ba.m_d = d;
		m_dgemm_nt_ba.m_rowsb = i1->m_weight;
		m_dgemm_nt_ba.m_colsa = w2;
		m_dgemm_nt_ba.m_colsb = w1;
		m_dgemm_nt_ba.m_ldb = i1->m_incb;
		m_dgemm_nt_ba.m_lda = k1w1;
		m_dgemm_nt_ba.m_ldc = i1->m_incc;
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_n_b_l3(
	double d, size_t w1, size_t w2, size_t k1w1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	---------------
	//	w   a  b      c
	//	w1  1  1      0
	//	w2  0  k1*w1  1  -->  c_i = a_p b_i%p
	//	---------------       sz(i) = w2, sz(p) = w1, sz(%) = k1
	//	                      [dgemv_n_b]
	//

	//	1. Minimize k2:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      1      0
	//	w2  0      k1*w1  1
	//	w3  k2*w1  0      k3*w2  -->  c_j#i = a_j$p b_i%p
	//	-----------------------       sz(i) = w2, sz(j) = w3,
	//	                              sz(p) = w1
	//	                              sz(#) = k3, sz($) = k2,
	//	                              sz(%) = k1
	//	                              [dgemm_nt_ab]
	//
	size_t k2_min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_incb != 0) continue;
		if(i->m_incc % w2 != 0) continue;
		if(i->m_inca % w1 != 0) continue;

		register size_t k2 = i->m_inca / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_nt_ab";
		m_kernelname = "dgemm_nt_ab";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemm_nt_ab;
		m_dgemm_nt_ab.m_d = d;
		m_dgemm_nt_ab.m_rowsa = i1->m_weight;
		m_dgemm_nt_ab.m_colsb = w2;
		m_dgemm_nt_ab.m_colsa = w1;
		m_dgemm_nt_ab.m_lda = i1->m_inca;
		m_dgemm_nt_ab.m_ldb = k1w1;
		m_dgemm_nt_ab.m_ldc = i1->m_incc;
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_t_a1_l3(
	double d, size_t w1, size_t w2, size_t k1, size_t k2w1) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	-----------------
	//	w   a      b   c
	//	w1  1      0   k1
	//	w2  k2*w1  1   0   -->  c_i# = a_p$i b_p
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz(#) = k1, sz($) = k2
	//	                        [dgemv_t_a]
	//

	//	1. Minimize k4:
	//	------------------------
	//	w   a      b      c
	//	w1  1      0      k1'*w3
	//	w2  k2*w1  1      0
	//	w3  0      k4*w2  1       -->  c_i#j = a_p$i b_j%p
	//	------------------------       sz(i) = w1, sz(j) = w3,
	//	                               sz(p) = w2
	//	                               sz(#) = k1', sz($) = k2,
	//	                               sz(%) = k4
	//	                               [dgemm_tt_ab]
	//
	size_t k4_min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 0 || i->m_incc != 1) continue;
		if(k1 % i->m_weight != 0) continue;
		if(i->m_incb % w2 != 0) continue;

		register size_t k4 = i->m_incb / w2;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_tt_ab";
		m_kernelname = "dgemm_tt_ab";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemm_tt_ab;
		m_dgemm_tt_ab.m_d = d;
		m_dgemm_tt_ab.m_rowsa = w1;
		m_dgemm_tt_ab.m_colsb = i1->m_weight;
		m_dgemm_tt_ab.m_colsa = w2;
		m_dgemm_tt_ab.m_lda = k2w1;
		m_dgemm_tt_ab.m_ldb = i1->m_incb;
		m_dgemm_tt_ab.m_ldc = k1;
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_t_a2_l3(
	double d, size_t w1, size_t w2, size_t k2w1, size_t k3) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	---------------
	//	w   a     b   c
	//	w1  1     0   1
	//	w2  k2w1  k3  0  -->  c_i = a_p$i b_p%
	//	---------------       sz(i) = w1, sz(p) = w2,
	//	                      sz($) = k2, sz(%) = k3
	//	                      [dgemv_t_a]
	//

	//	1. If k3 == 1, minimize k5:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      0      1
	//	w2  k2*w1  1      0
	//	w3  0      k5*w2  k6*w1  --> c_j#i = a_p$i b_j%p
	//	-----------------------      sz(i) = w1, sz(j) = w3,
	//	                             sz(p) = w2
	//	                             sz(#) = k6, sz($) = k2,
	//	                             sz(%) = k5
	//	                             [dgemm_nn_ba]
	//
	if(k3 == 1) {
		size_t k5_min = 0;
		list_iter i1 = m_list.end();
		for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
			if(i->m_inca != 0) continue;
			if(i->m_incc % w1 != 0) continue;
			if(i->m_incb % w2 != 0) continue;

			register size_t k5 = i->m_incb / w2;
			if(k5_min == 0 || k5_min > k5) {
				k5_min = k5; i1 = i;
			}
		}
		if(i1 != m_list.end()) {
			//~ std::cout << " dgemm_nn_ba";
			m_kernelname = "dgemm_nn_ba";
			i1->m_fn = &tod_contract2<N, M, K>::fn_dgemm_nn_ba;
			m_dgemm_nn_ba.m_d = d;
			m_dgemm_nn_ba.m_rowsb = i1->m_weight;
			m_dgemm_nn_ba.m_colsa = w1;
			m_dgemm_nn_ba.m_colsb = w2;
			m_dgemm_nn_ba.m_ldb = i1->m_incb;
			m_dgemm_nn_ba.m_lda = k2w1;
			m_dgemm_nn_ba.m_ldc = i1->m_incc;
			m_list.splice(m_list.end(), m_list, i1);
			return;
		}
	}

	//	2. Minimize k4:
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
	size_t k4_min = 0;
	list_iter i2 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 0 || i->m_incb != 1) continue;
		if(k3 % i->m_weight != 0) continue;
		if(i->m_incc % w1 != 0) continue;

		register size_t k4 = i->m_incc / w1;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i2 = i;
		}
	}
	if(i2 != m_list.end()) {
		//~ std::cout << " dgemm_tn_ba";
		m_kernelname = "dgemm_tn_ba";
		i2->m_fn = &tod_contract2<N, M, K>::fn_dgemm_tn_ba;
		m_dgemm_tn_ba.m_d = d;
		m_dgemm_tn_ba.m_rowsb = i2->m_weight;
		m_dgemm_tn_ba.m_colsa = w1;
		m_dgemm_tn_ba.m_colsb = w2;
		m_dgemm_tn_ba.m_ldb = k3;
		m_dgemm_tn_ba.m_lda = k2w1;
		m_dgemm_tn_ba.m_ldc = i2->m_incc;
		m_list.splice(m_list.end(), m_list, i2);
		return;
	}
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::match_dgemv_t_b_l3(
	double d, size_t w1, size_t w2, size_t k2w1, size_t k3) {

	typedef typename loop_list_t::iterator list_iter;

	//	Found pattern:
	//	----------------
	//	w   a   b      c
	//	w1  0   1      1
	//	w2  k3  k2w1   0  -->  c_i = a_p$ b_p%i
	//	----------------       sz(i) = w1, sz(p) = w2,
	//	                       sz($) = k3, sz(%) = k2
	//	                       [dgemv_t_b]
	//

	//	1. Minimize k4:
	//	-----------------------
	//	w   a       b     c
	//	w1  0       1     1
	//	w2  k3'*w3  k2w1  0
	//	w3  1       0     k4*w1  -->  c_j#i = a_p$j b_p%i
	//	-----------------------       sz(i) = w1, sz(j) = w3,
	//	                              sz(p) = w2
	//	                              sz(#) = k4, sz($) = k3',
	//	                              sz(%) = k2
	//	                              [dgemm_tn_ab]
	//
	size_t k4_min = 0;
	list_iter i1 = m_list.end();
	for(list_iter i = m_list.begin(); i != m_list.end(); i++) {
		if(i->m_inca != 1 || i->m_incb != 0) continue;
		if(k3 % i->m_weight != 0) continue;
		if(i->m_incc % w1 != 0) continue;

		register size_t k4 = i->m_incc / w1;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i1 = i;
		}
	}
	if(i1 != m_list.end()) {
		//~ std::cout << " dgemm_tn_ab";
		m_kernelname = "dgemm_tn_ab";
		i1->m_fn = &tod_contract2<N, M, K>::fn_dgemm_tn_ab;
		m_dgemm_tn_ab.m_d = d;
		m_dgemm_tn_ab.m_rowsa = i1->m_weight;
		m_dgemm_tn_ab.m_colsb = w1;
		m_dgemm_tn_ab.m_colsa = w2;
		m_dgemm_tn_ab.m_lda = k3;
		m_dgemm_tn_ab.m_ldb = k2w1;
		m_dgemm_tn_ab.m_ldc = i1->m_incc;
		m_list.splice(m_list.end(), m_list, i1);
		return;
	}
}


template<size_t N, size_t M, size_t K>
inline void tod_contract2<N, M, K>::loop_list_adapter::append(size_t weight,
	size_t inca, size_t incb, size_t incc) {
	m_list.push_back(loop_list_node(weight, inca, incb, incc));
}


template<size_t N, size_t M, size_t K>
inline void tod_contract2<N, M, K>::exec(
	loop_list_iterator_t &i, registers &r) {

	void (tod_contract2<N, M, K>::*fnptr)(registers&) = i->m_fn;

	if(fnptr == 0) fn_loop(i, r);
	else (this->*fnptr)(r);
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_loop(loop_list_iterator_t &i, registers &r) {

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


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_ddot(registers &r) {

	double d = m_ddot.m_d;
	size_t n = m_ddot.m_n;

//	tod_contract2<N, M, K>::start_timer("ddot");
	r.m_ptrc[0] += d * cblas_ddot(n, r.m_ptra, 1, r.m_ptrb, 1);
//	tod_contract2<N, M, K>::stop_timer("ddot");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_daxpy_a(registers &r) {

	double d = m_daxpy_a.m_d;
	size_t n = m_daxpy_a.m_n;
	size_t stepc = m_daxpy_a.m_stepc;

//	tod_contract2<N, M, K>::start_timer("daxpy_a");
	cblas_daxpy(n, r.m_ptrb[0] * d, r.m_ptra, 1, r.m_ptrc, stepc);
//	tod_contract2<N, M, K>::stop_timer("daxpy_a");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_daxpy_b(registers &r) {

	double d = m_daxpy_b.m_d;
	size_t n = m_daxpy_b.m_n;
	size_t stepc = m_daxpy_b.m_stepc;

//	tod_contract2<N, M, K>::start_timer("daxpy_b");
	cblas_daxpy(n, r.m_ptra[0] * d, r.m_ptrb, 1, r.m_ptrc, stepc);
//	tod_contract2<N, M, K>::stop_timer("daxpy_b");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemv_n_a(registers &r) {

	double d = m_dgemv_n_a.m_d;
	size_t rows = m_dgemv_n_a.m_rows;
	size_t cols = m_dgemv_n_a.m_cols;
	size_t lda = m_dgemv_n_a.m_lda;
	size_t stepb = m_dgemv_n_a.m_stepb;

//	tod_contract2<N, M, K>::start_timer("dgemv_n_a");
	cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, d, r.m_ptra, lda,
		r.m_ptrb, stepb, 1.0, r.m_ptrc, 1);
//	tod_contract2<N, M, K>::stop_timer("dgemv_n_a");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemv_t_a(registers &r) {

	double d = m_dgemv_t_a.m_d;
	size_t rows = m_dgemv_t_a.m_rows;
	size_t cols = m_dgemv_t_a.m_cols;
	size_t lda = m_dgemv_t_a.m_lda;
	size_t stepb = m_dgemv_t_a.m_stepb;
	size_t stepc = m_dgemv_t_a.m_stepc;

//	tod_contract2<N, M, K>::start_timer("dgemv_t_a");
	cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, d, r.m_ptra, lda,
		r.m_ptrb, stepb, 1.0, r.m_ptrc, stepc);
//	tod_contract2<N, M, K>::stop_timer("dgemv_t_a");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemv_n_b(registers &r) {

	double d = m_dgemv_n_b.m_d;
	size_t rows = m_dgemv_n_b.m_rows;
	size_t cols = m_dgemv_n_b.m_cols;
	size_t ldb = m_dgemv_n_b.m_ldb;
	size_t stepa = m_dgemv_n_b.m_stepa;

//	tod_contract2<N, M, K>::start_timer("dgemv_n_b");
	cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, d, r.m_ptrb, ldb,
		r.m_ptra, stepa, 1.0, r.m_ptrc, 1);
//	tod_contract2<N, M, K>::stop_timer("dgemv_n_b");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemv_t_b(registers &r) {

	double d = m_dgemv_t_b.m_d;
	size_t rows = m_dgemv_t_b.m_rows;
	size_t cols = m_dgemv_t_b.m_cols;
	size_t ldb = m_dgemv_t_b.m_ldb;
	size_t stepa = m_dgemv_t_b.m_stepa;
	size_t stepc = m_dgemv_t_b.m_stepc;

//	tod_contract2<N, M, K>::start_timer("dgemv_t_b");
	cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, d, r.m_ptrb, ldb,
		r.m_ptra, stepa, 1.0, r.m_ptrc, stepc);
//	tod_contract2<N, M, K>::stop_timer("dgemv_t_b");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_nt_ab(registers &r) {

	double d = m_dgemm_nt_ab.m_d;
	size_t rowsa = m_dgemm_nt_ab.m_rowsa;
	size_t colsb = m_dgemm_nt_ab.m_colsb;
	size_t colsa = m_dgemm_nt_ab.m_colsa;
	size_t lda = m_dgemm_nt_ab.m_lda;
	size_t ldb = m_dgemm_nt_ab.m_ldb;
	size_t ldc = m_dgemm_nt_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nt_ab");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		rowsa, colsb, colsa, d, r.m_ptra, lda, r.m_ptrb, ldb,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nt_ab");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_tn_ab(registers &r) {

	double d = m_dgemm_tn_ab.m_d;
	size_t rowsa = m_dgemm_tn_ab.m_rowsa;
	size_t colsb = m_dgemm_tn_ab.m_colsb;
	size_t colsa = m_dgemm_tn_ab.m_colsa;
	size_t lda = m_dgemm_tn_ab.m_lda;
	size_t ldb = m_dgemm_tn_ab.m_ldb;
	size_t ldc = m_dgemm_tn_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tn_ab");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		rowsa, colsb, colsa, d, r.m_ptra, lda, r.m_ptrb, ldb,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tn_ab");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_tt_ab(registers &r) {

	double d = m_dgemm_tt_ab.m_d;
	size_t rowsa = m_dgemm_tt_ab.m_rowsa;
	size_t colsb = m_dgemm_tt_ab.m_colsb;
	size_t colsa = m_dgemm_tt_ab.m_colsa;
	size_t lda = m_dgemm_tt_ab.m_lda;
	size_t ldb = m_dgemm_tt_ab.m_ldb;
	size_t ldc = m_dgemm_tt_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tt_ab");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
		rowsa, colsb, colsa, d, r.m_ptra, lda, r.m_ptrb, ldb,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tt_ab");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_nn_ba(registers &r) {

	double d = m_dgemm_nn_ba.m_d;
	size_t rowsb = m_dgemm_nn_ba.m_rowsb;
	size_t colsa = m_dgemm_nn_ba.m_colsa;
	size_t colsb = m_dgemm_nn_ba.m_colsb;
	size_t ldb = m_dgemm_nn_ba.m_ldb;
	size_t lda = m_dgemm_nn_ba.m_lda;
	size_t ldc = m_dgemm_nn_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nn_ba");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		rowsb, colsa, colsb, d, r.m_ptrb, ldb, r.m_ptra, lda,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nn_ba");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_nt_ba(registers &r) {

	double d = m_dgemm_nt_ba.m_d;
	size_t rowsb = m_dgemm_nt_ba.m_rowsb;
	size_t colsa = m_dgemm_nt_ba.m_colsa;
	size_t colsb = m_dgemm_nt_ba.m_colsb;
	size_t ldb = m_dgemm_nt_ba.m_ldb;
	size_t lda = m_dgemm_nt_ba.m_lda;
	size_t ldc = m_dgemm_nt_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nt_ba");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		rowsb, colsa, colsb, d, r.m_ptrb, ldb, r.m_ptra, lda,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nt_ba");
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::fn_dgemm_tn_ba(registers &r) {

	double d = m_dgemm_tn_ba.m_d;
	size_t rowsb = m_dgemm_tn_ba.m_rowsb;
	size_t colsa = m_dgemm_tn_ba.m_colsa;
	size_t colsb = m_dgemm_tn_ba.m_colsb;
	size_t ldb = m_dgemm_tn_ba.m_ldb;
	size_t lda = m_dgemm_tn_ba.m_lda;
	size_t ldc = m_dgemm_tn_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tn_ba");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		rowsb, colsa, colsb, d, r.m_ptrb, ldb, r.m_ptra, lda,
		1.0, r.m_ptrc, ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tn_ba");
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

