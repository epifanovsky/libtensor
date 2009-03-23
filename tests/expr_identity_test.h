#ifndef LIBTENSOR_EXPR_IDENTITY_TEST_H
#define LIBTENSOR_EXPR_IDENTITY_TEST_H

#include <libtest.h>
#include "expr_identity.h"

namespace libtensor {

/**	\brief Tests the libtensor::expr_identity class

	\ingroup libtensor_tests
**/
class expr_identity_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_IDENTITY_TEST_H

