#ifndef LIBTENSOR_TOD_DIAG_H
#define LIBTENSOR_TOD_DIAG_H

#include "../defs.h"
#include "../linalg.h"
#include "../not_implemented.h"
#include "../timings.h"
#include "../core/mask.h"
#include "../core/permutation.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "bad_dimensions.h"

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

private:
	tensor_i<N, double> &m_t; //!< Input %tensor
	mask<N> m_mask; //!< Diagonal mask.
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
		\param t Output %tensor.
	 **/
	void perform(tensor_i<N - M + 1, double> &t);

	/**	\brief Performs the operation, adds to the output
		\param t Output %tensor.
		\param c Coefficient.
	 **/
	void perform(tensor_i<N - M + 1, double> &t, double c);

private:
	/**	\brief Forms the %dimensions of the output or throws an
			exception if the input is incorrect
	 **/
	static dimensions<N - M + 1> mk_dims(
		const dimensions<N> &dims, const mask<N> &m);
};


template<size_t N, size_t M>
const char *tod_diag<N, M>::k_clazz = "tod_diag<N, M>";


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(tensor_i<N, double> &t, const mask<N> &m, double c) :
	m_t(t), m_mask(m), m_c(c), m_dims(mk_dims(t.get_dims(), m)) {

}


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(tensor_i<N, double> &t, const mask<N> &m,
	const permutation<N - M + 1> &p, double c) :
	m_t(t), m_mask(m), m_perm(p), m_c(c), m_dims(mk_dims(t.get_dims(), m)) {

	m_dims.permute(p);
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(tensor_i<N - M + 1, double> &t) {

	static const char *method = "perform(tensor_i<N - M + 1, double> &)";

	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"t");
	}

	tod_diag<N, M>::start_timer();

	size_t inc = 0, m = 0, sz = 0;
	const dimensions<N> &dimsa = m_t.get_dims();
	for(size_t i = 0; i < N; i++) {
		if(m_mask[i]) {
			if(m == 0) sz = dimsa[i];
			m++;
			inc += dimsa.get_increment(i);
		}
	}

	if(m != M) {
		tod_diag<N, M>::stop_timer();
		throw not_implemented(
			g_ns, k_clazz, method, __FILE__, __LINE__);
	}

	tensor_ctrl<N, double> ca(m_t);
	tensor_ctrl<N - M + 1, double> cb(t);
	const double *pa = ca.req_const_dataptr();
	double *pb = cb.req_dataptr();
	blas_dcopy(sz, pa, inc, pb, 1);
	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_diag<N, M>::stop_timer();
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(tensor_i<N - M + 1, double> &t, double c) {

	static const char *method =
		"perform(tensor_i<N - M + 1, double> &, double)";

	throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N, size_t M>
dimensions<N - M + 1> tod_diag<N, M>::mk_dims(
	const dimensions<N> &dims, const mask<N> &msk) {

	static const char *method =
		"mk_dims(const dimensions<N> &, const mask<N> &)";

	index<N - M + 1> i1, i2;

	size_t m = 0, d = 0, j = 0;
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
	return dimensions<N - M + 1>(index_range<N - M + 1>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIAG_H
