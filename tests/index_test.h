#ifndef LIBTENSOR_INDEX_TEST_H
#define LIBTENSOR_INDEX_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::index class

	\ingroup libtensor_tests
**/
class index_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);

	//!	Tests the index::less() method
	void test_less() throw(libtest::test_exception);

	//!	Tests the operator<<
	void test_print() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_INDEX_TEST_H

