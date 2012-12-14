#ifndef LIBTENSOR_TOD_CUDA_COPY_H
#define LIBTENSOR_TOD_CUDA_COPY_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "kernels/cuda_kern_copy_generic.h"
//#include "loop_list_add.h"
//#include "loop_list_copy.h"
//#include "tod_additive.h"
#include <libtensor/core/bad_dimensions.h>

namespace libtensor {

/**	\brief Makes a copy of a %tensor, scales or permutes %tensor elements
		if necessary
	\tparam N Tensor order.

	This operation makes a scaled and permuted copy of a %tensor.
	The result can replace or be added to the output %tensor.


	<b>Examples</b>

	Plain copy:
	\code
	dense_tensor_i<2, double> &t1(...), &t2(...);
	tod_cuda_copy<2> cp(t1);
	cp.perform(t2); // Copies the elements of t1 to t2
	\endcode

	Scaled copy:
	\code
	dense_tensor_i<2, double> &t1(...), &t2(...);
	tod_cuda_copy<2> cp(t1, 0.5);
	cp.perform(t2); // Copies the elements of t1 multiplied by 0.5 to t2
	\endcode

	Permuted copy:
	\code
	dense_tensor_i<2, double> &t1(...), &t2(...);
	permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
	tod_cuda_copy<2> cp(t1, perm);
	cp.perform(t2); // Copies transposed t1 to t2
	\endcode

	Permuted and scaled copy:
	\code
	dense_tensor_i<2, double> &t1(...), &t2(...);
	permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
	tod_cuda_copy<2> cp(t1, perm, 0.5);
	cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
	\endcode

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_cuda_copy :
	public timings< tod_cuda_copy<N> > {

public:
	static const char *k_clazz; //!< Class name

private:
	dense_tensor_i<N, double> &m_ta; //!< Source %tensor
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
	tod_cuda_copy(dense_tensor_i<N, double> &ta, double c = 1.0);

	/**	\brief Prepares the permute & copy operation
		\param ta Source %tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	 **/
	tod_cuda_copy(dense_tensor_i<N, double> &ta, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~tod_cuda_copy() { }

	//@}


	//!	\name Implementation of libtensor::tod_additive<N>
	//@{

	virtual void prefetch();

	//!	\copydoc tod_additive<N>::perform(dense_tensor_i<N, double>&)
	virtual void perform(dense_tensor_i<N, double> &t);

	//!	\copydoc tod_additive<N>::perform(dense_tensor_i<N, double>&, double)
	virtual void perform(dense_tensor_i<N, double> &t, double c);

	//@}

private:
	/**	\brief Creates the dimensions of the output using an input
			%tensor and a permutation of indexes
	 **/
	static dimensions<N> mk_dimsb(dense_tensor_i<N, double> &ta,
		const permutation<N> &perm);

	void do_perform(dense_tensor_i<N, double> &t, double c);
//
//	template<typename Base>
//	void build_loop(typename Base::list_t &loop, const dimensions<N> &dimsa,
//		const permutation<N> &perma, const dimensions<N> &dimsb);

};


template<size_t N>
const char *tod_cuda_copy<N>::k_clazz = "tod_cuda_copy<N>";


template<size_t N>
tod_cuda_copy<N>::tod_cuda_copy(dense_tensor_i<N, double> &ta, double c) :
	m_ta(ta), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {

}


template<size_t N>
tod_cuda_copy<N>::tod_cuda_copy(dense_tensor_i<N, double> &ta, const permutation<N> &p,
	double c) : m_ta(ta), m_perm(p), m_c(c), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N>
void tod_cuda_copy<N>::prefetch() {

	dense_tensor_ctrl<N, double>(m_ta).req_prefetch();
}


template<size_t N>
void tod_cuda_copy<N>::perform(dense_tensor_i<N, double> &tb) {

	static const char *method = "perform(dense_tensor_i<N, double>&)";

	if(!tb.get_dims().equals(m_dimsb)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tb");
	}

	do_perform(tb, 0);
}


template<size_t N>
void tod_cuda_copy<N>::perform(dense_tensor_i<N, double> &tb, double c) {

	static const char *method = "perform(dense_tensor_i<N, double>&, double)";

	if(!tb.get_dims().equals(m_dimsb)) {
//		std::cout << "\n m_dimsb = " << m_dimsb << "\n tb dims = " << tb.get_dims();
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,	"tb");
	}
	if(c == 0) return;

	do_perform(tb, c);
}


template<size_t N>
dimensions<N> tod_cuda_copy<N>::mk_dimsb(dense_tensor_i<N, double> &ta,
	const permutation<N> &perm) {

	dimensions<N> dims(ta.get_dims());
	dims.permute(perm);
	return dims;
}


template<size_t N>
void tod_cuda_copy<N>::do_perform(dense_tensor_i<N, double> &tb, double c) {


	tod_cuda_copy<N>::start_timer();

	try {

	dense_tensor_ctrl<N, double> ca(m_ta), cb(tb);
	ca.req_prefetch();
	cb.req_prefetch();

	const dimensions<N> &dimsa = m_ta.get_dims();

	const double *pa = ca.req_const_dataptr();
	double *pb = cb.req_dataptr();

	cuda_kern_copy_generic *kern = cuda_kern_copy_generic::match(pa, pb, dimsa, m_perm, m_c, c);
	tod_cuda_copy<N>::start_timer(kern->get_name());
	kern->run();

	tod_cuda_copy<N>::stop_timer(kern->get_name());
	delete kern; kern = 0;


	ca.ret_const_dataptr(pa);
	cb.ret_dataptr(pb);

	} catch(...) {
		tod_cuda_copy<N>::stop_timer();
		throw;
	}
	tod_cuda_copy<N>::stop_timer();
}


//template<size_t N> template<typename Base>
//void tod_cuda_copy<N>::build_loop(typename Base::list_t &loop,
//	const dimensions<N> &dimsa, const permutation<N> &perma,
//	const dimensions<N> &dimsb) {
//
//	typedef typename Base::iterator_t iterator_t;
//	typedef typename Base::node node_t;
//
//	sequence<N, size_t> map(0);
//	for(register size_t i = 0; i < N; i++) map[i] = i;
//	perma.apply(map);
//
//	//
//	//	Go over indexes in B and connect them with indexes in A
//	//	trying to glue together consecutive indexes
//	//
//	for(size_t idxb = 0; idxb < N;) {
//		size_t len = 1;
//		size_t idxa = map[idxb];
//		do {
//			len *= dimsa.get_dim(idxa);
//			idxa++; idxb++;
//		} while(idxb < N && map[idxb] == idxa);
//
//		iterator_t inode = loop.insert(loop.end(), node_t(len));
//		inode->stepa(0) = dimsa.get_increment(idxa - 1);
//		inode->stepb(0) = dimsb.get_increment(idxb - 1);
//	}
//}


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_H
