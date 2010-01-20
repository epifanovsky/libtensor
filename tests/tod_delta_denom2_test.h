#ifndef LIBTENSOR_TOD_DELTA_DENOM2_TEST_H
#define LIBTENSOR_TOD_DELTA_DENOM2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_delta_denom2 class

	\ingroup libtensor_tests
**/
class tod_delta_denom2_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_operation(size_t ni, size_t na)
		throw(libtest::test_exception);
	void test_exceptions() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_DELTA_DENOM2_H

