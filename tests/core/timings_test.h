#ifndef LIBTENSOR_TIMINGS_TEST_H
#define LIBTENSOR_TIMINGS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::timer class

	\ingroup libtensor_tests
**/
class timings_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TIMINGS_TEST_H

