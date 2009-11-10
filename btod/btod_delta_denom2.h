#ifndef LIBTENSOR_BTOD_DELTA_DENOM2_H
#define LIBTENSOR_BTOD_DELTA_DENOM2_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"

namespace libtensor {

class btod_delta_denom2 {
private:
	block_tensor_i<2, double> &m_dov; //!< The delta matrix
	double m_thresh; //!< Zero threshold
	double m_min; //!< The minimum absolute value of the denominator

public:
	/**	\brief Creates and initializes the operation
		\param dov The delta matrix
			\f$ \Delta_{\sigma_i i \sigma_a a} \f$
	 **/
	btod_delta_denom2(block_tensor_i<2, double> &dov,
		double thresh = 0.0);

	/**	\brief Performs the operation
	 **/
	void perform(block_tensor_i<4, double> &t) throw(exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DELTA_DENOM2_H
