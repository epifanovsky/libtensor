#ifndef LIBTENSOR_TOD_SET_TEST_H
#define LIBTENSOR_TOD_SET_TEST_H

#include <libtest.h>
#include "tod_set.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_set class

	\ingroup libtensor_tests
**/
class tod_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_TEST_H

