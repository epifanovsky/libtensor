#ifndef LIBTENSOR_TOD_SET_ELEM_TEST_H
#define LIBTENSOR_TOD_SET_ELEM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_set_elem class

	\ingroup libtensor_tests
**/
class tod_set_elem_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_ELEM_TEST_H
