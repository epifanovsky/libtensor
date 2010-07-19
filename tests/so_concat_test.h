#ifndef LIBTENSOR_SO_CONCAT_TEST_H
#define LIBTENSOR_SO_CONCAT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::so_concat class

	\ingroup libtensor_tests
**/
class so_concat_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_1a() throw(libtest::test_exception);
	void test_1b() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5a() throw(libtest::test_exception);
	void test_5b() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SO_CONCAT_TEST_H
