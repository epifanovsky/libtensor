#ifndef LIBTENSOR_SE_PERM_TEST_H
#define LIBTENSOR_SE_PERM_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::se_perm class

	\ingroup libtensor_tests
 **/
class se_perm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_sym_ab_ba() throw(libtest::test_exception);
	void test_asym_ab_ba() throw(libtest::test_exception);
	void test_sym_abc_bca() throw(libtest::test_exception);
	void test_asym_abc_bca() throw(libtest::test_exception);
	void test_sym_abcd_badc() throw(libtest::test_exception);
	void test_asym_abcd_badc() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_TEST_H

