#ifndef LIBTENSOR_BTOD_MULT_TEST_H
#define LIBTENSOR_BTOD_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_mult class

	\ingroup libtensor_tests
**/
class btod_mult_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_TEST_H
