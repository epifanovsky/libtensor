#ifndef LIBTENSOR_SYMM_TEST_H
#define	LIBTENSOR_SYMM_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::symm function

	\ingroup libtensor_tests
**/
class symm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_symm_contr_tt_1() throw(libtest::test_exception);
	void test_symm_contr_ee_1() throw(libtest::test_exception);
	void test_asymm_contr_tt_1() throw(libtest::test_exception);
	void test_asymm_contr_tt_2() throw(libtest::test_exception);
	void test_asymm_contr_tt_3() throw(libtest::test_exception);
	void test_asymm_contr_tt_4() throw(libtest::test_exception);
	void test_asymm_contr_ee_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SYMM_TEST_H