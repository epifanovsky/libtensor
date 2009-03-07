#ifndef LIBTENSOR_TOD_SUM_TEST_H
#define LIBTENSOR_TOD_SUM_TEST_H

#include <libtest.h>
#include "tod_sum.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_sum class

	\ingroup libtensor_tests
**/
class tod_sum_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SUM_TEST_H

