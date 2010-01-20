#ifndef LIBTENSOR_TOD_RANDOM_TEST_H
#define LIBTENSOR_TOD_RANDOM_TEST_H

#include <cmath>
#include <cstdlib>
#include <libtest/unit_test.h>
#include <libtensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_random class

	\ingroup libtensor_tests
**/
class tod_random_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_TEST_H

