#ifndef LIBTENSOR_BTOD_CONTRACT2_TEST_H
#define LIBTENSOR_BTOD_CONTRACT2_TEST_H

#include <libtest.h>
#include "btod_contract2.h"


namespace libtensor {

/**	\brief Tests the libtensor::btod_contract2 class

	\ingroup libtensor_tests
**/
class btod_contract2_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_TEST_H
