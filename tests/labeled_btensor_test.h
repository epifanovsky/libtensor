#ifndef LIBTENSOR_LABELED_BTENSOR_TEST_H
#define	LIBTENSOR_LABELED_BTENSOR_TEST_H

#include <libtest.h>
#include "labeled_btensor.h"

namespace libtensor {

/**	\brief Tests the libtensor::labeled_btensor class

	\ingroup libtensor_tests
**/
class labeled_btensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_label() throw(libtest::test_exception);
	void test_expr() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_TEST_H

