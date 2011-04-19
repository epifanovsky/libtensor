#ifndef LIBTENSOR_TOD_SET_DIAG_H
#define LIBTENSOR_TOD_SET_DIAG_H

#include "../defs.h"
#include "../exception.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Assigns the diagonal elements of a %tensor to a value
	\tparam N Tensor order.

	This operation sets the diagonal elements of a %tensor to a value
	without affecting all the off-diagonal elements. The dimensions of the
	%tensor must all be the same.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_set_diag {
public:
	static const char *k_clazz; //!< Class name

private:
	double m_v; //!< Value

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param v Tensor element value (default 0.0).
	 **/
	tod_set_diag(double v = 0.0);

	//@}

	//!	\name Operation
	//@{

	/**	\brief Performs the operation
	 **/
	void perform(tensor_i<N, double> &t) throw(exception);

	//@}
};


template<size_t N>
const char *tod_set_diag<N>::k_clazz = "tod_set_diag<N>";


template<size_t N>
tod_set_diag<N>::tod_set_diag(double v) : m_v(v) {

}


template<size_t N>
void tod_set_diag<N>::perform(tensor_i<N, double> &t) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&)";

	const dimensions<N> &dims = t.get_dims();
	size_t n = dims[0];
	for(size_t i = 1; i < N; i++) {
		if(dims[i] != n) {
			throw bad_dimensions(g_ns, k_clazz, method, __FILE__,
				__LINE__, "t.");
		}
	}

	size_t inc = 0;
	for(size_t i = 0; i < N; i++) inc += dims.get_increment(i);

	tensor_ctrl<N, double> ctrl(t);
	double *d = ctrl.req_dataptr();
	for(size_t i = 0; i < n; i++) d[i*inc] = m_v;
	ctrl.ret_dataptr(d);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_DIAG_H
