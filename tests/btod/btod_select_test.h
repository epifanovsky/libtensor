#ifndef LIBTENSOR_BTOD_SELECT_TEST_H
#define LIBTENSOR_BTOD_SELECT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_select class

	\ingroup libtensor_tests_btod
**/
class btod_select_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<typename ComparePolicy>
	void test_1(size_t n) throw(libtest::test_exception);
	template<typename ComparePolicy>
	void test_2(size_t n) throw(libtest::test_exception);
	template<typename ComparePolicy>
	void test_3(size_t n, bool symm) throw(libtest::test_exception);
	template<typename ComparePolicy>
	void test_4(size_t n, bool symm) throw(libtest::test_exception);
	template<typename ComparePolicy>
	void test_5(size_t n) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_TEST_H
