#ifndef LIBTENSOR_TOD_BTCONV_TEST_H
#define LIBTENSOR_TOD_BTCONV_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_btconv class

	\ingroup libtensor_tests
**/
class tod_btconv_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_TEST_H
