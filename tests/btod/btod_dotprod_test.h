#ifndef LIBTENSOR_BTOD_DOTPROD_TEST_H
#define LIBTENSOR_BTOD_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_dotprod class

	\ingroup libtensor_tests
**/
class btod_dotprod_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6() throw(libtest::test_exception);
	void test_7() throw(libtest::test_exception);
	void test_8() throw(libtest::test_exception);
	void test_9() throw(libtest::test_exception);
	void test_10() throw(libtest::test_exception);
	void test_11() throw(libtest::test_exception);
	void test_12() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_TEST_H
