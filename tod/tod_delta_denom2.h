#ifndef LIBTENSOR_TOD_DELTA_DENOM2_H
#define LIBTENSOR_TOD_DELTA_DENOM2_H

#include "defs.h"
#include "exception.h"
#include "core/tensor_i.h"

namespace libtensor {

/**	\brief Applies the MP2 delta denominator to a two-particle operator

	Given two delta matrices \f$ \Delta^{(1)}_{\sigma_i i \sigma_a a} =
	f_{\sigma_i i \sigma_i i}-f_{\sigma_a a \sigma_a a} \f$ and
	\f$ \Delta^{(2)}_{\sigma_j j \sigma_b b} \f$, as well as amplitudes
	\f$ t_{\sigma_i i \sigma_j j \sigma_a a \sigma_b b} \f$, this operation
	performs the division:
	\f[ t_{\sigma_i i \sigma_j j \sigma_a a \sigma_b b} =
		\frac{t_{\sigma_i i \sigma_j j \sigma_a a \sigma_b b}}
		{\Delta_{\sigma_i i \sigma_a a} +
			\Delta_{\sigma_j j \sigma_b b}} \f]

	The threshold parameter controls the minimum absolute value of the
	denominator. If the denominator is smaller than the threshold, the
	amplitude is divided by the threshold instead of the sum of delta
	elements. The default value of the threshold is 0.0.

	Along with modifying the amplitudes tensor, the operation also
	produces the smallest denominator absolute value. The user can obtain
	that value by calling get_min() after executing the operation.

	\see libtensor::tod_make_delta

	\ingroup libtensor_tod
**/
class tod_delta_denom2 {
private:
	tensor_i<2, double> &m_dov1; //!< Delta matrix 1
	tensor_i<2, double> &m_dov2; //!< Delta matrix 2
	double m_thresh; //!< Zero threshold
	double m_min; //!< The minimum absolute value of the denominator

public:
	/**	\brief Creates and initializes the operation
		\param dov1 First delta matrix
			\f$ \Delta_{\sigma_i i \sigma_a a} \f$
		\param dov2 Second delta matrix
			\f$ \Delta_{\sigma_j j \sigma_b b} \f$
	**/
	tod_delta_denom2(tensor_i<2, double> &dov1,
		tensor_i<2, double> &dov2,
		double thresh = 0.0);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_delta_denom2();

	/**	\brief Requests the prefetch of input parameters
	**/
	virtual void prefetch() throw(exception);

	/**	\brief Performs the operation
	**/
	virtual void perform(tensor_i<4, double> &t)
		throw(exception);

	/**	\brief Returns the minimum denominator absolute value
	**/
	double get_min();

private:
	void inner_step(size_t i, size_t j, size_t na, size_t nb,
		const double *p_dov1_i, const double *p_dov2_j, double *p_t);
};

inline double tod_delta_denom2::get_min() {
	return m_min;
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_DELTA_DENOM2_H

