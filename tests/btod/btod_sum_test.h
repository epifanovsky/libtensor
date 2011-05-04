#ifndef LIBTENSOR_BTOD_SUM_TEST_H
#define LIBTENSOR_BTOD_SUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_sum class

	\ingroup libtensor_tests_btod
**/
class btod_sum_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6(bool do_add) throw(libtest::test_exception);
	void test_7() throw(libtest::test_exception);
	void test_8() throw(libtest::test_exception);
	void test_9a() throw(libtest::test_exception);
	void test_9b() throw(libtest::test_exception);
	void test_10a() throw(libtest::test_exception);
	void test_10b() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_TEST_H
