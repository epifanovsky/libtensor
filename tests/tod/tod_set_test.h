#ifndef LIBTENSOR_TOD_SET_TEST_H
#define LIBTENSOR_TOD_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_set class

	\ingroup libtensor_tests_tod
**/
class tod_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_TEST_H

