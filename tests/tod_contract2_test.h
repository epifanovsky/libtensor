#ifndef LIBTENSOR_TOD_CONTRACT2_TEST_H
#define LIBTENSOR_TOD_CONTRACT2_TEST_H

#include <libtest.h>
#include "tod_contract2.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_contract2 class

	\ingroup libtensor_tests
**/
class tod_contract2_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_TEST_H

