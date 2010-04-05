#ifndef LIBTENSOR_BTOD_TRACE_TEST_H
#define LIBTENSOR_BTOD_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_trace class

	\ingroup libtensor_tests
 **/
class btod_trace_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_zero_1() throw(libtest::test_exception);
	void test_nosym_1() throw(libtest::test_exception);
	void test_nosym_1_sp() throw(libtest::test_exception);
	void test_nosym_2() throw(libtest::test_exception);
	void test_nosym_3() throw(libtest::test_exception);
	void test_nosym_4() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRACE_TEST_H
