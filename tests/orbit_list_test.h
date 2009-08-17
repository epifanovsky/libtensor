#ifndef LIBTENSOR_ORBIT_LIST_TEST_H
#define LIBTENSOR_ORBIT_LIST_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::orbit_list class

	\ingroup libtensor_tests
**/
class orbit_list_test : public libtest::unit_test {
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
};

} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_TEST_H
