#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "loop_list_add.h"
#include "loop_list_copy.h"
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
	public loop_list_add,
	public loop_list_copy,
	public tod_additive<N>,
	public timings< tod_copy<N> > {

public:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<N, double> &m_ta; //!< Source %tensor
	permutation<N> m_perm; //!< Permutation of elements
	double m_c; //!< Scaling coefficient
	dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Prepares the copy operation
		\param ta Source %tensor.
		\param c Coefficient.
	 **/
	tod_copy(tensor_i<N, double> &ta, double c = 1.0);

	/**	\brief Prepares the permute & copy operation
		\param ta Source %tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	 **/
	tod_copy(tensor_i<N, double> &ta, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~tod_copy() { }

	//@}


	//!	\name Implementation of libtensor::tod_additive<N>
	//@{

	//!	\copydoc tod_additive<N>::perform(tensor_i<N, double>&)
	virtual void perform(tensor_i<N, double> &t) throw(exception);

	//!	\copydoc tod_additive<N>::perform(tensor_i<N, double>&, double)
	virtual void perform(tensor_i<N, double> &t, double c) throw(exception);

	//@}

private:
	/**	\brief Creates the dimensions of the output using an input
			%tensor and a permutation of indexes
	 **/
	static dimensions<N> mk_dimsb(tensor_i<N, double> &ta,
		const permutation<N> &perm);

	template<typename Base>
	void do_perform_copy(tensor_i<N, double> &t, double c);

	template<typename Base>
	void build_loop(typename Base::list_t &loop, const dimensions<N> &dimsa,
		const permutation<N> &perma, const dimensions<N> &dimsb);

};


template<size_t N>
const char *tod_copy<N>::k_clazz = "tod_copy<N>";


template<size_t N>
tod_copy<N>::tod_copy(tensor_i<N, double> &ta, double c) :
	m_ta(ta), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {

}


template<size_t N>
tod_copy<N>::tod_copy(tensor_i<N, double> &ta, const permutation<N> &p,
	double c) : m_ta(ta), m_perm(p), m_c(c), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tb) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&)";

	if(!tb.get_dims().equals(m_dimsb)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tb");
	}

	do_perform_copy<loop_list_copy>(tb, 1.0);
}


template<size_t N>
void tod_copy<N>::perform(tensor_i<N, double> &tb, double c) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&, double)";

	if(!tb.get_dims().equals(m_dimsb)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tb");
	}
	if(c == 0) return;

	do_perform_copy<loop_list_add>(tb, c);
}


template<size_t N>
dimensions<N> tod_copy<N>::mk_dimsb(tensor_i<N, double> &ta,
	const permutation<N> &perm) {

	dimensions<N> dims(ta.get_dims());
	dims.permute(perm);
	return dims;
}


template<size_t N> template<typename Base>
void tod_copy<N>::do_perform_copy(tensor_i<N, double> &tb, double c) {

	typedef typename Base::list_t list_t;
	typedef typename Base::registers registers_t;
	typedef typename Base::node node_t;

	tod_copy<N>::start_timer();

	try {

	tensor_ctrl<N, double> ca(m_ta), cb(tb);
	ca.req_prefetch();
	cb.req_prefetch();

	const dimensions<N> &dimsa = m_ta.get_dims();
	const dimensions<N> &dimsb = tb.get_dims();

	list_t loop;
	build_loop<Base>(loop, dimsa, m_perm, dimsb);

	const double *pa = ca.req_const_dataptr();
	double *pb = cb.req_dataptr();

	registers_t r;
	r.m_ptra[0] = pa;
	r.m_ptrb[0] = pb;
#ifdef LIBTENSOR_DEBUG
	r.m_ptra_end[0] = pa + dimsa.get_size();
	r.m_ptrb_end[0] = pb + dimsb.get_size();
#endif // LIBTENSOR_DEBUG

	Base::run_loop(loop, r, m_c * c);

	ca.ret_dataptr(pa);
	cb.ret_dataptr(pb);

	} catch(...) {
		tod_copy<N>::stop_timer();
		throw;
	}
	tod_copy<N>::stop_timer();
}


template<size_t N> template<typename Base>
void tod_copy<N>::build_loop(typename Base::list_t &loop,
	const dimensions<N> &dimsa, const permutation<N> &perma,
	const dimensions<N> &dimsb) {

	typedef typename Base::iterator_t iterator_t;
	typedef typename Base::node node_t;

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

		iterator_t inode = loop.insert(loop.end(), node_t(len));
		inode->stepa(0) = dimsa.get_increment(idxa - 1);
		inode->stepb(0) = dimsb.get_increment(idxb - 1);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_H
