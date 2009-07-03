#ifndef LIBTENSOR_TOD_DOTPROD_TEST_H
#define LIBTENSOR_TOD_DOTPROD_TEST_H

#include <cmath>
#include <cstdlib>
#include <libtest.h>
#include <libtensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_dotprod class

	\ingroup libtensor_tests
**/
class tod_dotprod_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_TEST_H

