#ifndef LIBTENSOR_EXPR_LITERAL_TEST_H
#define LIBTENSOR_EXPR_LITERAL_TEST_H

#include <libtest.h>
#include "expr_literal.h"

namespace libtensor {

/**	\brief Tests the libtensor::expr_literal class

	\ingroup libtensor_tests
**/
class expr_literal_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_LITERAL_TEST_H

