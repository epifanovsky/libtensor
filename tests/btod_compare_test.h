#ifndef LIBTENSOR_BTOD_COMPARE_TEST_H
#define LIBTENSOR_BTOD_COMPARE_TEST_H

#include <libtest.h>
#include <libtensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_compare class

	\ingroup libtensor_tests
**/
class btod_compare_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	/**	\brief Tests if an exception is throws when the tensors have
			different dimensions
	**/
	void test_exc() throw(libtest::test_exception);

	/**	\brief Tests the operation
	**/
	void test_operation() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_COMPARE_TEST_H

