#ifndef LIBTENSOR_TOD_MULT1_H
#define LIBTENSOR_TOD_MULT1_H

#include "../defs.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Element-wise multiplication and division
	\tparam N Tensor order.

	The operation multiplies or divides two tensors element by element.

	\f[ a_i = a_i b_i \qquad a_i = \frac{a_i}{b_i} \f]
	\f[ a_i = a_i + c a_i b_i \qquad a_i = a_i + c \frac{a_i}{b_i} \f]

	The numerator and the result are the same %tensor. Both tensors must
	have the same %dimensions or an exception will be thrown. When
	the division is requested, no checks are performed to ensure that
	the denominator is non-zero.

	\ingroup libtensor
 **/
template<size_t N>
class tod_mult1 : public timings< tod_mult1<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<N, double> &m_tb; //!< Second argument
	bool m_recip; //!< Reciprocal (multiplication by 1/bi)

public:
	/**	\brief Creates the operation
		\param tb Second argument.
		\param recip \c false (default) sets up multiplication and
			\c true sets up element-wise division.
	 **/
	tod_mult1(tensor_i<N, double> &tb, bool recip = false);

	/**	\brief Performs the operation, replaces the output.
		\param ta Tensor A.
	 **/
	void perform(tensor_i<N, double> &ta);

	/**	\brief Performs the operation, adds to the output.
		\param ta Tensor A.
		\param c Coefficient.
	 **/
	void perform(tensor_i<N, double> &ta, double c);
};


template<size_t N>
const char *tod_mult1<N>::k_clazz = "tod_mult1<N>";


template<size_t N>
tod_mult1<N>::tod_mult1(tensor_i<N, double> &tb, bool recip) :
	m_tb(tb), m_recip(recip) {
}


template<size_t N>
void tod_mult1<N>::perform(tensor_i<N, double> &ta) {

	static const char *method = "perform(tensor_i<N, double>&)";

	if(!m_tb.get_dims().equals(ta.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"ta");
	}

	tod_mult1<N>::start_timer();

	tensor_ctrl<N, double> ca(ta), cb(m_tb);
	double *pa = ca.req_dataptr();
	const double *pb = cb.req_const_dataptr();

	size_t sz = ta.get_dims().get_size();
	if(m_recip) {
		for(size_t i = 0; i < sz; i++) {
			pa[i] /= pb[i];
		}
	} else {
		for(size_t i = 0; i < sz; i++) {
			pa[i] *= pb[i];
		}
	}

	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_mult1<N>::stop_timer();
}


template<size_t N>
void tod_mult1<N>::perform(tensor_i<N, double> &ta, double c) {

	static const char *method = "perform(tensor_i<N, double>&, double)";

	if(!m_tb.get_dims().equals(ta.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"ta");
	}

	tod_mult1<N>::start_timer();

	tensor_ctrl<N, double> ca(ta), cb(m_tb);
	double *pa = ca.req_dataptr();
	const double *pb = cb.req_const_dataptr();

	size_t sz = ta.get_dims().get_size();
	if(m_recip) {
		for(size_t i = 0; i < sz; i++) {
			pa[i] += c * pa[i] / pb[i];
		}
	} else {
		for(size_t i = 0; i < sz; i++) {
			pa[i] += c * pa[i] * pb[i];
		}
	}

	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_mult1<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT1_H
