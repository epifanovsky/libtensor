#ifndef LIBTENSOR_TOD_DIAG_H
#define LIBTENSOR_TOD_DIAG_H

#include "../defs.h"
#include "../linalg/linalg.h"
#include "../not_implemented.h"
#include "../timings.h"
#include "../core/mask.h"
#include "../core/permutation.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "bad_dimensions.h"
#include "processor.h"

namespace libtensor {


/**	\brief Extracts a general diagonal from a %tensor
	\tparam N Tensor order.
	\tparam M Diagonal order.

	Extracts a general multi-dimensional diagonal from a %tensor. The
	diagonal to extract is specified by a %mask, unmasked indexes remain
	intact. The order of the result is (n-m+1), where n is the order of
	the original %tensor, m is the order of the diagonal.

	The order of indexes in the result is the same as in the argument with
	the exception of the collapsed diagonal. The diagonal's index in the
	result correspond to the first its index in the argument, for example:
	\f[ c_i = a_{ii} \qquad c_{ip} = a_{iip} \qquad c_{ip} = a_{ipi} \f]
	The specified permutation may be applied to the result to alter the
	order of the indexes.

	A coefficient (default 1.0) is specified to scale the elements along
	with the extraction of the diagonal.

	If the number of set bits in the %mask is not equal to M, the %mask
	is incorrect, which causes a \c bad_parameter exception upon the
	creation of the operation. If the %dimensions of the output %tensor
	are wrong, the \c bad_dimensions exception is thrown.

	\ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_diag : public timings< tod_diag<N, M> > {
public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the source %tensor
	static const size_t k_orderb =
		N - M + 1; //!< Order of the destination %tensor

private:
	struct registers {
		const double *m_ptra;
		double *m_ptrb;
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
		loop_list_node() :
			m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb) :
			m_weight(weight), m_inca(inca), m_incb(incb),
			m_op(NULL) { }
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

	class op_dcopy : public processor_op_i_t, public timings<op_dcopy> {
	public:
		static const char *k_clazz; //!< Class name
	private:
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_dcopy(size_t len, size_t inca, size_t incb, double c) :
			m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_daxpy : public processor_op_i_t, public timings<op_daxpy> {
	public:
		static const char *k_clazz; //!< Class name
	private:
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_daxpy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

private:
	tensor_i<N, double> &m_t; //!< Input %tensor
	mask<N> m_mask; //!< Diagonal mask
	permutation<N - M + 1> m_perm; //!< Permutation of the result
	double m_c; //!< Scaling coefficient
	dimensions<N - M + 1> m_dims; //!< Dimensions of the result

public:
	/**	\brief Creates the operation
		\param t Input %tensor.
		\param m Diagonal mask.
		\param c Scaling coefficient (default 1.0).
	 **/
	tod_diag(tensor_i<N, double> &t, const mask<N> &m, double c = 1.0);

	/**	\brief Creates the operation
		\param t Input %tensor.
		\param m Diagonal mask.
		\param p Permutation of result.
		\param c Scaling coefficient (default 1.0)
	 **/
	tod_diag(tensor_i<N, double> &t, const mask<N> &m,
		const permutation<N - M + 1> &p, double c = 1.0);

	/**	\brief Performs the operation, replaces the output
		\param tb Output %tensor.
	 **/
	void perform(tensor_i<k_orderb, double> &tb);

	/**	\brief Performs the operation, adds to the output
		\param tb Output %tensor.
		\param c Coefficient.
	 **/
	void perform(tensor_i<k_orderb, double> &tb, double c);

private:
	/**	\brief Forms the %dimensions of the output or throws an
			exception if the input is incorrect
	 **/
	static dimensions<N - M + 1> mk_dims(
		const dimensions<N> &dims, const mask<N> &msk);

	/**	\brief Forms the loop and executes the operation
	 **/
	template<typename CoreOp>
	void do_perform(tensor_i<k_orderb, double> &tb, double c);

	/**	\brief Builds the nested loop list
	 **/
	template<typename CoreOp>
	void build_list(
		loop_list_t &list, tensor_i<k_orderb, double> &tb, double c);

	/**	\brief Cleans the nested loop list
	 **/
	void clean_list(loop_list_t &list);
};


template<size_t N, size_t M>
const char *tod_diag<N, M>::k_clazz = "tod_diag<N, M>";

template<size_t N, size_t M>
const char *tod_diag<N, M>::op_dcopy::k_clazz = "tod_diag<N, M>::op_dcopy";

template<size_t N, size_t M>
const char *tod_diag<N, M>::op_daxpy::k_clazz = "tod_diag<N, M>::op_daxpy";


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(tensor_i<N, double> &t, const mask<N> &m, double c) :

	m_t(t), m_mask(m), m_c(c), m_dims(mk_dims(t.get_dims(), m_mask)) {

}


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(tensor_i<N, double> &t, const mask<N> &m,
	const permutation<N - M + 1> &p, double c) :

	m_t(t), m_mask(m), m_perm(p), m_c(c),
	m_dims(mk_dims(t.get_dims(), m_mask)) {

	m_dims.permute(p);
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(tensor_i<k_orderb, double> &tb) {

	static const char *method = "perform(tensor_i<N - M + 1, double> &)";

	if(!tb.get_dims().equals(m_dims)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"t");
	}

	do_perform<op_dcopy>(tb, 1.0);
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(tensor_i<k_orderb, double> &tb, double c) {

	static const char *method =
		"perform(tensor_i<N - M + 1, double> &, double)";

	if(!tb.get_dims().equals(m_dims)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"t");
	}

	do_perform<op_daxpy>(tb, c);
}


template<size_t N, size_t M>
dimensions<N - M + 1> tod_diag<N, M>::mk_dims(const dimensions<N> &dims,
	const mask<N> &msk) {

	static const char *method =
		"mk_dims(const dimensions<N> &, const mask<N>&)";

	//	Compute output dimensions
	//
	index<k_orderb> i1, i2;

	size_t m = 0, j = 0;
	size_t d = 0;
	bool bad_dims = false;
	for(size_t i = 0; i < N; i++) {
		if(msk[i]) {
			m++;
			if(d == 0) {
				d = dims[i];
				i2[j++] = d - 1;
			} else {
				bad_dims = bad_dims || d != dims[i];
			}
		} else {
			if(!bad_dims) i2[j++] = dims[i] - 1;
		}
	}
	if(m != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"m");
	}
	if(bad_dims) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"t");
	}
	return dimensions<k_orderb>(index_range<k_orderb>(i1, i2));
}


template<size_t N, size_t M> template<typename CoreOp>
void tod_diag<N, M>::do_perform(tensor_i<k_orderb, double> &tb, double c) {

	static const char *method =
		"do_perform(tensor_i<N - M + 1, double>&, double)";

	tod_diag<N, M>::start_timer();

	tensor_ctrl<k_ordera, double> ca(m_t);
	tensor_ctrl<k_orderb, double> cb(tb);
	const double *pa = ca.req_const_dataptr();
	double *pb = cb.req_dataptr();

	loop_list_t lst;
	build_list<CoreOp>(lst, tb, c * m_c);

	registers regs;
	regs.m_ptra = pa;
	regs.m_ptrb = pb;

	try {
		processor_t proc(lst, regs);
		proc.process_next();
	} catch(...) {
		clean_list(lst);
		throw;
	}

	clean_list(lst);

	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_diag<N, M>::stop_timer();
}


template<size_t N, size_t M> template<typename CoreOp>
void tod_diag<N, M>::build_list(
	loop_list_t &list, tensor_i<k_orderb, double> &tb, double c) {

	static const char *method = "build_list(loop_list_t&, "
		"tensor_i<N - M + 1, double>&, double)";

	const dimensions<k_ordera> &dimsa = m_t.get_dims();
	const dimensions<k_orderb> &dimsb = tb.get_dims();

	//	Mapping of unpermuted indexes in b to permuted ones
	//
	size_t ib[k_orderb];
	for(size_t i = 0; i < k_orderb; i++) ib[i] = i;
	permutation<k_orderb> pinv(m_perm, true);
	pinv.apply(ib);

	//	Loop over the indexes and build the list
	//
	try { // bad_alloc

	typename loop_list_t::iterator poscore = list.end();
	bool diag_done = false;
	size_t iboffs = 0;
	for(size_t pos = 0; pos < N; pos++) {

		size_t inca = 0, incb = 0, len = 0;

		if(m_mask[pos]) {

			if(diag_done) {
				iboffs++;
				continue;
			}

			//	Compute the stride on the diagonal
			//
			for(size_t j = pos; j < N; j++)
				if(m_mask[j]) inca += dimsa.get_increment(j);
			incb = dimsb.get_increment(ib[pos]);
			len = dimsa.get_dim(pos);
			diag_done = true;

		} else {

			//	Compute the stride off the diagonal
			//	concatenating indexes if possible
			//
			len = 1;
			size_t ibpos = ib[pos - iboffs];
			while(pos < N && !m_mask[pos] && ibpos == ib[pos - iboffs]) {

				len *= dimsa.get_dim(pos);
				pos++;
				ibpos++;
			}
			pos--; ibpos--;
			inca = dimsa.get_increment(pos);
			incb = dimsb.get_increment(ibpos);
		}


		typename loop_list_t::iterator it = list.insert(
			list.end(), loop_list_node(len, inca, incb));

		//	Make the loop with incb the last
		//
		if(incb == 1 && poscore == list.end()) {
			it->m_op = new CoreOp(len, inca, incb, c);
			poscore = it;
		} else {
			it->m_op = new op_loop(len, inca, incb);
		}
	}

	list.splice(list.end(), list, poscore);

	} catch(std::bad_alloc &e) {

		clean_list(list);
		throw out_of_memory(
			g_ns, k_clazz, method, __FILE__, __LINE__, e.what());
	}
}


template<size_t N, size_t M>
void tod_diag<N, M>::clean_list(loop_list_t& lst) {

	for(typename loop_list_t::iterator i = lst.begin();
		i != lst.end(); i++) {

		delete i->m_op; i->m_op = 0;
	}
}


template<size_t N, size_t M>
void tod_diag<N, M>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {

	const double *ptra = regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}
}


template<size_t N, size_t M>
void tod_diag<N, M>::op_dcopy::exec(processor_t &proc, registers &regs)
	throw(exception) {

	if(m_len == 0) return;

	op_dcopy::start_timer();
	linalg::i_i(m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
	if(m_c != 1.0) linalg::i_x(m_len, m_c, regs.m_ptrb, m_incb);
	op_dcopy::stop_timer();
}


template<size_t N, size_t M>
void tod_diag<N, M>::op_daxpy::exec(processor_t &proc, registers &regs)
	throw(exception) {

	if(m_len == 0) return;

	op_daxpy::start_timer();
	linalg::i_i_x(m_len, regs.m_ptra, m_inca, m_c, regs.m_ptrb, m_incb);
	op_daxpy::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIAG_H
