#ifndef LIBTENSOR_TOD_SOLVE_TEST_H
#define LIBTENSOR_TOD_SOLVE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_solve class

	\ingroup libtensor_tests
 **/
class tod_solve_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SOLVE_TEST_H
