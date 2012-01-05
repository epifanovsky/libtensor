#ifndef LIBTENSOR_TOD_SET_ELEM_H
#define LIBTENSOR_TOD_SET_ELEM_H

#include "../defs.h"
#include "../exception.h"
#include "../core/abs_index.h"
#include "../core/index.h"
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>

namespace libtensor {

/**	\brief Assigns a value to a single %tensor element
	\tparam N Tensor order.

	This operation allows access to individual %tensor elements addressed
	by their %index. It is useful to set one or two elements to a particular
	value, but too slow to routinely work with %tensors.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_set_elem {
public:
	//!	\name Operation
	//@{

	/**	\brief Assigns the element of a tensor a value
		\param t Tensor.
	 **/
	void perform(dense_tensor_i<N, double> &t, const index<N> &idx, double d);

	//@}
};


template<size_t N>
void tod_set_elem<N>::perform(dense_tensor_i<N, double> &t, const index<N> &idx,
	double d) {

	abs_index<N> aidx(idx, t.get_dims());
	dense_tensor_ctrl<N, double> ctrl(t);
	double *p = ctrl.req_dataptr();
	p[aidx.get_abs_index()] = d;
	ctrl.ret_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_ELEM_H
