#ifndef LIBTENSOR_SYMMETRY_TEST_H
#define LIBTENSOR_SYMMETRY_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::symmetry class

	\ingroup libtensor_tests
**/
class symmetry_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_TEST_H

