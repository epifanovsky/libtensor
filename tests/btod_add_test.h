#ifndef LIBTENSOR_BTOD_ADD_TEST_H
#define LIBTENSOR_BTOD_ADD_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_add class

	\ingroup libtensor_tests
**/
class btod_add_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	/**	\brief Tests if exceptions are thrown when the tensors have
			different dimensions
	**/
	void test_exc() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_TEST_H
