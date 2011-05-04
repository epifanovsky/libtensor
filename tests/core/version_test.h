#ifndef LIBTENSOR_VERSION_TEST_H
#define LIBTENSOR_VERSION_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::version class

	\ingroup libtensor_tests_core
**/
class version_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_VERSION_TEST_H
