#ifndef LIBTENSOR_CONTRACT2_2_2I_TEST_H
#define LIBTENSOR_CONTRACT2_2_2I_TEST_H

#include <libtest.h>
#include "contract2_2_2i.h"

namespace libtensor {

/**	\brief Tests the libtensor::contract2_2_2i class

	\ingroup libtensor_tests
**/
class contract2_2_2i_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_2I_TEST_H

