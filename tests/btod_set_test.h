#ifndef LIBTENSOR_BTOD_SET_TEST_H
#define LIBTENSOR_BTOD_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_set class

	\ingroup libtensor_tests
**/
class btod_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_TEST_H
