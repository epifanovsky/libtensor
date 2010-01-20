#ifndef LIBTENSOR_MASK_TEST_H
#define LIBTENSOR_MASK_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::mask class

	\ingroup libtensor_tests
**/
class mask_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_MASK_TEST_H
