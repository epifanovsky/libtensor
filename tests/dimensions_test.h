#ifndef LIBTENSOR_DIMENSIONS_TEST_H
#define LIBTENSOR_DIMENSIONS_TEST_H

#include <libtest.h>
#include "dimensions.h"

namespace libtensor {

/**	\brief Tests the libtensor::dimensions class

	\ingroup libtensor_tests
**/
class dimensions_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);

	//!	Tests the dimensions::inc_index() method
	void test_inc_index() throw(libtest::test_exception);

	//!	Tests the dimensions::abs_index() method
	void test_abs_index() throw(libtest::test_exception);

	//!	Tests the comparison operations for dimensions
	void test_comp() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_TEST_H

