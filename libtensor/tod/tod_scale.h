#ifndef LIBTENSOR_TOD_SCALE_H
#define LIBTENSOR_TOD_SCALE_H

#include "../defs.h"
#include "../linalg/linalg.h"
#include "../timings.h"
#include "../dense_tensor/dense_tensor_i.h"
#include "../core/tensor_ctrl.h"

namespace libtensor {


/**	\brief Scales a %tensor by a constant
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_scale : public timings< tod_scale<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	dense_tensor_i<N, double> &m_t; //!< Tensor
	double m_c; //!< Scaling coefficient

public:
	/**	\brief Initializes the operation
		\param t Tensor.
		\param c Scaling coefficient
	 **/
	tod_scale(dense_tensor_i<N, double> &t, double c) : m_t(t), m_c(c) { }

	/**	\brief Performs the operation
	 **/
	void perform();

private:
	tod_scale(const tod_scale<N> &);
	tod_scale<N> &operator=(const tod_scale<N> &);
};


template<size_t N>
const char *tod_scale<N>::k_clazz = "tod_scale<N>";


template<size_t N>
void tod_scale<N>::perform() {

	tod_scale<N>::start_timer();

	tensor_ctrl<N, double> ctrl(m_t);
	double *p = ctrl.req_dataptr();
	linalg::i_x(m_t.get_dims().get_size(), m_c, p, 1);
	ctrl.ret_dataptr(p); p = 0;

	tod_scale<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCALE_H
