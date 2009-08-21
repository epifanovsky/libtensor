#ifndef LIBTENSOR_BTOD_RANDOM_TEST_H
#define LIBTENSOR_BTOD_RANDOM_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_random class

	\ingroup libtensor_tests
**/
class btod_random_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_TEST_H
