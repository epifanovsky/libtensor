#ifndef LIBTENSOR_LINALG_TEST_BASE_H
#define LIBTENSOR_LINALG_TEST_BASE_H

#include <cmath>
#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Base test class for linalg tests

	\ingroup libtensor_tests_linalg
 **/
class linalg_test_base : public libtest::unit_test {
protected:
	bool cmp(double diff, double ref) {

		const double k_thresh = 1e-12;

		if(fabs(ref) > 1.0) return fabs(diff) < fabs(ref) * k_thresh;
		else return fabs(diff) < k_thresh;
	}
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_TEST_BASE_H
