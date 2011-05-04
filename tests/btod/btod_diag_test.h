#ifndef LIBTENSOR_BTOD_DIAG_TEST_H
#define LIBTENSOR_BTOD_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_diag class

	\ingroup libtensor_tests_btod
**/
class btod_diag_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_zero_1() throw(libtest::test_exception);
	void test_zero_2() throw(libtest::test_exception);
	void test_nosym_1(bool add) throw(libtest::test_exception);
	void test_nosym_2(bool add) throw(libtest::test_exception);
	void test_nosym_3(bool add) throw(libtest::test_exception);
	void test_nosym_4(bool add) throw(libtest::test_exception);
	void test_sym_1(bool add) throw(libtest::test_exception);
	void test_sym_2(bool add) throw(libtest::test_exception);
	void test_sym_3(bool add) throw(libtest::test_exception);
	void test_sym_4(bool add) throw(libtest::test_exception);
	void test_sym_5(bool add) throw(libtest::test_exception);
	void test_sym_6(bool add) throw(libtest::test_exception);
	void test_sym_7(bool add) throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_TEST_H
