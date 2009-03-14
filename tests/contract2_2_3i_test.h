#ifndef LIBTENSOR_CONTRACT2_2_3I_TEST_H
#define LIBTENSOR_CONTRACT2_2_3I_TEST_H

#include <libtest.h>
#include "contract2_2_3i.h"

namespace libtensor {

/**	\brief Tests the libtensor::contract2_2_3i class

	\ingroup libtensor_tests
**/
class contract2_2_3i_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_3I_TEST_H

