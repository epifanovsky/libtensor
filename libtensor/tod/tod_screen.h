#ifndef LIBTENSOR_TOD_SCREEN_H
#define LIBTENSOR_TOD_SCREEN_H

#include <cmath>
#include "../defs.h"
#include "../timings.h"
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>

namespace libtensor {


/**	\brief Screens a %tensor for a certain element value
	\tparam N Tensor order.

	The operation goes over %tensor elements and searches for a given
	element value within a threshold. If requested, the values that match
	within the threshold are replaced with the exact value.

	The return value indicates whether one or more elements matched.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_screen : public timings< tod_screen<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	double m_a; //!< Value
	bool m_replace; //!< Whether to replace found elements
	double m_thresh; //!< Equality threshold

public:
	/**	\brief Initializes the operation
		\param a Element value (default 0.0).
		\param replace Whether to replace found elements
			(default false).
		\param thresh Threshold (default 0.0 -- exact match).
	 **/
	tod_screen(double a = 0.0, bool replace = false, double thresh = 0.0) :
		m_a(a), m_replace(replace), m_thresh(fabs(thresh)) { }

	/**	\brief Performs the operation
		\param t Tensor.
		\return True if match is found, false otherwise.
	 **/
	bool perform(dense_tensor_i<N, double> &t);

private:
	tod_screen(const tod_screen<N>&);
	const tod_screen<N> &operator=(const tod_screen<N>&);
};


template<size_t N>
const char *tod_screen<N>::k_clazz = "tod_screen<N>";


template<size_t N>
bool tod_screen<N>::perform(dense_tensor_i<N, double> &t) {

	dense_tensor_ctrl<N, double> ctrl(t);

	bool ret = false;

	if(m_replace) {

		size_t sz = t.get_dims().get_size();
		double *p = ctrl.req_dataptr();

		for(register size_t i = 0; i < sz; i++) {
			if(fabs(p[i] - m_a) < m_thresh) {
				p[i] = m_a;
				ret = true;
			}
		}

		ctrl.ret_dataptr(p);

	} else {

		size_t sz = t.get_dims().get_size();
		const double *p = ctrl.req_const_dataptr();

		for(register size_t i = 0; i < sz; i++) {
			if(fabs(p[i] - m_a) < m_thresh) ret = true;
		}

		ctrl.ret_dataptr(p);

	}

	return ret;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCREEN_H
