#ifndef LIBTENSOR_EXPRESSION_TEST_H
#define LIBTENSOR_EXPRESSION_TEST_H

#include <libtest.h>
#include "index.h"

namespace libtensor {

/**	\brief Tests tensor expressions

	\ingroup libtensor_tests
**/
class expression_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_EXPRESSION_TEST_H

