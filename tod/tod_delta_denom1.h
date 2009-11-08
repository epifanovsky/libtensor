#ifndef LIBTENSOR_TOD_DELTA_DENOM1_H
#define LIBTENSOR_TOD_DELTA_DENOM1_H

#include "defs.h"
#include "exception.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"

namespace libtensor {


/**	\brief Applies the MP2 delta denominator to a one-particle operator

	Given the delta matrix \f$ \Delta_{ia} = f_{ii}-f_{aa} \f$ and
	amplitudes \f$ t_{ia} \f$, this operation performs the division:
	\f[ t_{\sigma_i i \sigma_a a} = \frac{t_{\sigma_i i \sigma_a a}}
		{\Delta_{\sigma_i i \sigma_a a}} \delta_{\sigma_i \sigma_a} \f]

	The threshold parameter controls the minimum absolute value of the
	denominator. If the denominator is smaller than the threshold, the
	amplitude is divided by the threshold instead of the value of the delta
	matrix element. The default value of the threshold is 0.0.

	Along with modifying the amplitudes tensor, the operation also
	produces the smallest denominator absolute value. The user can obtain
	that value by calling get_min() after executing the operation.

	\see libtenosr::tod_mkdelta

	\ingroup libtensor_tod
**/
class tod_delta_denom1 {
private:
	tensor_i<2, double> &m_dov; //!< The delta matrix
	double m_thresh; //!< Zero threshold
	double m_min; //!< The minimum absolute value of the denominator

public:
	/**	\brief Creates and initializes the operation
		\param dov The delta matrix
			\f$ \Delta_{\sigma_i i \sigma_a a} \f$
	 **/
	tod_delta_denom1(tensor_i<2, double> &dov,
		double thresh = 0.0);

	/**	\brief Requests the prefetch of input parameters
	 **/
	void prefetch() throw(exception);

	/**	\brief Performs the operation
	 **/
	void perform(tensor_i<2, double> &t) throw(exception);

	/**	\brief Returns the minimum denominator absolute value
	 **/
	double get_min();
};


inline double tod_delta_denom1::get_min() {

	return m_min;
}


} // namespace libtensor

#endif // LIBTENOSR_TOD_DELTA_DENOM1_H
