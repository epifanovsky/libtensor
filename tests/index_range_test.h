#ifndef LIBTENSOR_INDEX_RANGE_TEST_H
#define LIBTENSOR_INDEX_RANGE_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::index_range class

	\ingroup libtensor_tests
**/
class index_range_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_INDEX_RANGE_TEST_H

