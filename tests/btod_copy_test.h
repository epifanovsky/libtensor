#ifndef LIBTENSOR_BTOD_COPY_TEST_H
#define LIBTENSOR_BTOD_COPY_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_copy class

	\ingroup libtensor_tests
**/
class btod_copy_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_TEST_H
