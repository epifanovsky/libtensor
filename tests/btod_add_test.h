#ifndef LIBTENSOR_BTOD_ADD_TEST_H
#define LIBTENSOR_BTOD_ADD_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_add class

	\ingroup libtensor_tests
 **/
class btod_add_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(double ca1, double ca2) throw(libtest::test_exception);
	void test_2(double ca1, double ca2, double cs)
		throw(libtest::test_exception);
	void test_3(double ca1, double ca2) throw(libtest::test_exception);
	void test_4(double ca1, double ca2, double ca3, double ca4)
		throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6() throw(libtest::test_exception);

	/**	\brief Tests if exceptions are thrown when the tensors have
			different dimensions
	**/
	void test_exc() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_TEST_H
