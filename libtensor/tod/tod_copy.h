#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "loop_list_copy.h"
#include "processor.h"
#include "tod_additive.h"
#include "bad_dimensions.h"

namespace libtensor {

/**	\brief Makes a copy of a %tensor, scales or permutes %tensor elements
		if necessary
	\tparam N Tensor order.

	This operation makes a scaled and permuted copy of a %tensor.
	The result can replace or be added to the output %tensor.

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
class tod_copy :
	public tod_additive<N>,
	public timings< tod_copy<N> >,
	public loop_list_copy {

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
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_dcopy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);

		static const char* k_clazz;
	};

	class op_daxpy : public processor_op_i_t, public timings<op_daxpy> {
	private:
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_daxpy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);

		static const char* k_clazz;
	};

private:
	tensor_i<N, double> &m_ta; //!< Source %tensor
	tensor_ctrl<N, double> m_tctrl; //!< Source %tensor control
	double m_c; //!< Scaling coefficient
	permutation<N> m_perm; //!< Permutation of elements
	dimensions<N> m_dimsb; //!< Dimensions of output %tensor

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
	static dimensions<N> mk_dimsb(tensor_i<N, double> &ta,
		const permutation<N> &perm);

	template<typename CoreOp>
	void do_perform(tensor_i<N, double> &t, double c) throw(exception);

	void do_perform_copy(tensor_i<N, double> &t, double c);

	template<typename List, typename Node>
	void build_loop(List &loop, const dimensions<N> &dimsa,
		const permutation<N> &perma, const dimensions<N> &dimsb);

	template<typename CoreOp>
	void build_list( loop_list_t &list, const dimensions<N> &dima,
		const permutation<N> &perma, const dimensions<N> &dimb,
		const double c) throw(out_of_memory);

	void clean_list( loop_list_t &list );

};

template<size_t N>
const char *tod_copy<N>::k_clazz = "tod_copy<N>";
template<size_t N>
const char *tod_copy<N>::op_dcopy::k_clazz = "tod_copy<N>::op_dcopy";
template<size_t N>
const char *tod_copy<N>::op_daxpy::k_clazz = "tod_copy<N>::op_daxpy";

template<size_t N>
inline tod_copy<N>::tod_copy(tensor_i<N, double> &t, double c)
	: m_ta(t), m_tctrl(t), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {
}

template<size_t N>
inline tod_copy<N>::tod_copy(tensor_i<N, double> &t,
	const permutation<N> &perm, double c)
	: m_ta(t), m_tctrl(t), m_c(c), m_perm(perm), m_dimsb(mk_dimsb(m_ta, m_perm)) {
}

template<size_t N>
tod_copy<N>::~tod_copy() {
}

template<size_t N>
inline void tod_copy<N>::prefetch() throw(exception) {
	m_tctrl.req_prefetch();
}

template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tb) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&)";

	if(!tb.get_dims().equals(m_dimsb)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tb");
	}

	do_perform_copy(tb, 1.0);
//	do_perform<op_dcopy>(tdst, 1.0);
}

template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tdst, double c)
	throw(exception) {
	do_perform<op_daxpy>(tdst, c);
}

template<size_t N>
dimensions<N> tod_copy<N>::mk_dimsb(tensor_i<N, double> &ta,
	const permutation<N> &perm) {

	dimensions<N> dims(ta.get_dims());
	dims.permute(perm);
	return dims;
}

template<size_t N> template<typename CoreOp>
void tod_copy<N>::do_perform(tensor_i<N, double> &tdst, double c)
	throw(exception) {

	static const char *method = "do_perform(tensor_i<N, double>&, double)";
	tod_copy<N>::start_timer();

	dimensions<N> dims(m_ta.get_dims()); dims.permute(m_perm);
	if(dims != tdst.get_dims()) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incorrect dimensions of the output tensor.");
	}

	tensor_ctrl<N, double> tctrl_dst(tdst);
	const double *psrc = m_tctrl.req_const_dataptr();
	double *pdst = tctrl_dst.req_dataptr();

//	permutation<N> inv_perm(m_perm);
//	inv_perm.invert();
//	size_t ib[N];
//	for(size_t i = 0; i < N; i++) ib[i] = i;
//	inv_perm.apply(ib);

	loop_list_t lst;
	build_list<CoreOp>( lst, m_ta.get_dims(), m_perm, tdst.get_dims(), c*m_c );

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
		clean_list(lst);
		throw;
	}

	clean_list(lst);

	m_tctrl.ret_dataptr(psrc);
	tctrl_dst.ret_dataptr(pdst);

	tod_copy<N>::stop_timer();
}


template<size_t N>
void tod_copy<N>::do_perform_copy(tensor_i<N, double> &tb, double c) {

	typedef loop_list_copy::list_t list_t;
	typedef loop_list_copy::registers registers_t;
	typedef loop_list_copy::node node_t;

	tod_copy<N>::start_timer();

	try {

	tensor_ctrl<N, double> ca(m_ta), cb(tb);
	ca.req_prefetch();
	cb.req_prefetch();

	const dimensions<N> &dimsa = m_ta.get_dims();
	const dimensions<N> &dimsb = tb.get_dims();

	const double *pa = ca.req_const_dataptr();
	double *pb = cb.req_dataptr();

	list_t loop;
	registers_t r;
	r.m_ptra = pa;
	r.m_ptrb = pb;
#ifdef LIBTENSOR_DEBUG
	r.m_ptra_end = pa + dimsa.get_size();
	r.m_ptrb_end = pb + dimsb.get_size();
#endif // LIBTENSOR_DEBUG

	build_loop<list_t, node_t>(loop, dimsa, m_perm, dimsb);
	loop_list_copy::run_loop(loop, r, m_c * c);

	ca.ret_dataptr(pa);
	cb.ret_dataptr(pb);

	} catch(...) {
		tod_copy<N>::stop_timer();
		throw;
	}
	tod_copy<N>::stop_timer();
}

template<size_t N> template<typename List, typename Node>
void tod_copy<N>::build_loop(List &loop, const dimensions<N> &dimsa,
	const permutation<N> &perma, const dimensions<N> &dimsb) {

	size_t map[N];
	for(register size_t i = 0; i < N; i++) map[i] = i;
	perma.apply(map);

	//
	//	Go over indexes in B and connect them with indexes in A
	//	trying to glue together consecutive indexes
	//
	for(size_t idxb = 0; idxb < N;) {
		size_t len = 1;
		size_t idxa = map[idxb];
		do {
			len *= dimsa.get_dim(idxa);
			idxa++; idxb++;
		} while(idxb < N && map[idxb] == idxa);

		loop.push_back(Node(len, dimsa.get_increment(idxa - 1),
			dimsb.get_increment(idxb - 1)));
	}
}


template<size_t N> template<typename CoreOp>
void tod_copy<N>::build_list( loop_list_t &list, const dimensions<N> &da,
	const permutation<N> &pa, const dimensions<N> &db, const double c ) throw(out_of_memory)
{
	size_t ia[N];
	for (size_t i=0; i<N; i++) ia[i]=i;
	pa.apply(N,ia);

	// loop over all indices and build the list
	size_t pos=0;
	try {
		typename loop_list_t::iterator posa=list.end(), posb=list.end();
		while ( pos < N ) {
			size_t len=1;
			size_t iapos=ia[pos];
			while (pos<N) {
				len*=da.get_dim(iapos);
				pos++; iapos++;
				if ( ia[pos]!=iapos ) break;
			}

			size_t inca=da.get_increment(iapos-1);
			size_t incb=db.get_increment(pos-1);

			typename loop_list_t::iterator it
				= list.insert(list.end(),loop_list_node(len,inca,incb));

			// we know that in the last loop incb=1 !!!
			if (inca==1) {
				if (inca==incb)	{
					it->m_op=new CoreOp(len,inca,incb,c);
				}
				else { posa=it; }
			}
			else {
				if (incb==1) { posb=it; }
				else { it->m_op=new op_loop(len,inca,incb); }
			}
		}

		if ( posa!=posb ) {
			if ( posa->m_weight > posb->m_weight ) {
				posa->m_op=new CoreOp(posa->m_weight,posa->m_inca,posa->m_incb,c);
				posb->m_op=new op_loop(posb->m_weight,posb->m_inca,posb->m_incb);
				list.splice(list.end(),list,posa);
			}
			else {
				posa->m_op=new op_loop(posa->m_weight,posa->m_inca,posa->m_incb);
				posb->m_op=new CoreOp(posb->m_weight,posb->m_inca,posb->m_incb,c);
				list.splice(posb,list,posa);
			}
		}
	} catch ( std::bad_alloc& e ) {
		clean_list(list);
		throw out_of_memory("libtensor",k_clazz,
			"build_list(loop_list_t&,const dimensions<N>&,"
			"const permutation<N>&,const dimensions<N>&)",
			__FILE__,__LINE__,e.what());
	}
}

template<size_t N>
void tod_copy<N>::clean_list( loop_list_t& lst ) {
	for(typename loop_list_t::iterator i = lst.begin();
		i != lst.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}
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

