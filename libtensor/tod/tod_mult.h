#ifndef LIBTENSOR_TOD_MULT_H
#define TOD_MULT_H

#include "../defs.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Element-wise multiplication and division
	\tparam N Tensor order.

	The operation multiplies or divides two tensors element by element.
	Both arguments and result must have the same %dimensions or an exception
	will be thrown. When the division is requested, no checks are performed
	to ensure that the denominator is non-zero.

	\ingroup libtensor
 **/
template<size_t N>
class tod_mult : public timings< tod_mult<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<N, double> &m_ta; //!< First argument
	tensor_i<N, double> &m_tb; //!< Second argument
	bool m_recip; //!< Reciprocal (multiplication by 1/bi)

public:
	/**	\brief Creates the operation
		\param ta First argument.
		\param tb Second argument.
		\param recip \c false (default) sets up multiplication and
			\c true sets up element-wise division.
	 **/
	tod_mult(tensor_i<N, double> &ta, tensor_i<N, double> &tb,
		bool recip = false);

	/**	\brief Performs the operation, replaces the output.
		\param tc Output %tensor.
	 **/
	void perform(tensor_i<N, double> &tc);

	/**	\brief Performs the operation, adds to the output.
		\param tc Output %tensor.
		\param c Coefficient.
	 **/
	void perform(tensor_i<N, double> &tc, double c);
};


template<size_t N>
const char *tod_mult<N>::k_clazz = "tod_mult<N>";


template<size_t N>
tod_mult<N>::tod_mult(
	tensor_i<N, double> &ta, tensor_i<N, double> &tb, bool recip) :

	m_ta(ta), m_tb(tb), m_recip(recip) {

	static const char *method =
		"tod_mult(tensor_i<N, double>&, tensor_i<N, double>&, bool)";

	if(!ta.get_dims().equals(tb.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"ta,tb");
	}

}


template<size_t N>
void tod_mult<N>::perform(tensor_i<N, double> &tc) {

	static const char *method = "perform(tensor_i<N, double>&)";

	if(!m_ta.get_dims().equals(tc.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tc");
	}

	tod_mult<N>::start_timer();

	tensor_ctrl<N, double> ca(m_ta), cb(m_tb), cc(tc);
	const double *pa = ca.req_const_dataptr();
	const double *pb = cb.req_const_dataptr();
	double *pc = cc.req_dataptr();

	size_t sz = tc.get_dims().get_size();
	if(m_recip) {
		for(size_t i = 0; i < sz; i++) {
			pc[i] = pa[i] / pb[i];
		}
	} else {
		for(size_t i = 0; i < sz; i++) {
			pc[i] = pa[i] * pb[i];
		}
	}

	cc.ret_dataptr(pc); pc = 0;
	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_mult<N>::stop_timer();
}


template<size_t N>
void tod_mult<N>::perform(tensor_i<N, double> &tc, double c) {

	static const char *method = "perform(tensor_i<N, double>&, double)";

	if(!m_ta.get_dims().equals(tc.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"tc");
	}

	tod_mult<N>::start_timer();

	tensor_ctrl<N, double> ca(m_ta), cb(m_tb), cc(tc);
	const double *pa = ca.req_const_dataptr();
	const double *pb = cb.req_const_dataptr();
	double *pc = cc.req_dataptr();

	size_t sz = tc.get_dims().get_size();
	if(m_recip) {
		for(size_t i = 0; i < sz; i++) {
			pc[i] += c * pa[i] / pb[i];
		}
	} else {
		for(size_t i = 0; i < sz; i++) {
			pc[i] += c * pa[i] * pb[i];
		}
	}

	cc.ret_dataptr(pc); pc = 0;
	cb.ret_dataptr(pb); pb = 0;
	ca.ret_dataptr(pa); pa = 0;

	tod_mult<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT_H
