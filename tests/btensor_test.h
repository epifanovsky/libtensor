#ifndef LIBTENSOR_BTENSOR_TEST_H
#define LIBTENSOR_BTENSOR_TEST_H

#include <libtest.h>

namespace libtensor {


/**	\brief Tests the libtensor::btensor class

	\ingroup libtensor_tests
**/
class btensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_TEST_H

