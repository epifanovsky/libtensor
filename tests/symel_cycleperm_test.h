#ifndef LIBTENSOR_SYMEL_CYCLEPERM_TEST_H
#define LIBTENSOR_SYMEL_CYCLEPERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::symel_cycleperm class

	\ingroup libtensor_tests
**/
class symel_cycleperm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_equals_1() throw(libtest::test_exception);
	void test_equals_2() throw(libtest::test_exception);
	void test_equals_3() throw(libtest::test_exception);
	void test_permute_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SYMEL_CYCLEPERM_TEST_H
