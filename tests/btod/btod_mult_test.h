#ifndef LIBTENSOR_BTOD_MULT_TEST_H
#define LIBTENSOR_BTOD_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_mult class

	\ingroup libtensor_tests_btod
**/
class btod_mult_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(bool recip, bool doadd) throw(libtest::test_exception);
	void test_2(bool recip, bool doadd) throw(libtest::test_exception);
	void test_3(bool recip, bool doadd) throw(libtest::test_exception);
	void test_4(bool recip, bool doadd) throw(libtest::test_exception);
	void test_5(bool symm1, bool symm2) throw(libtest::test_exception);
	void test_6(bool symm1, bool symm2) throw(libtest::test_exception);
	void test_7(bool label, bool part, bool samesym,
			bool recip, bool add) throw(libtest::test_exception);
	void test_8a(bool label, bool part) throw(libtest::test_exception);
	void test_8b(bool label, bool part) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_TEST_H
