#ifndef LIBTENSOR_SO_APPLY_IMPL_PERM_TEST_H
#define LIBTENSOR_SO_APPLY_IMPL_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::so_apply_impl_perm class

	\ingroup libtensor_tests_sym
 **/
class so_apply_impl_perm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(bool is_asym, bool sign) throw(libtest::test_exception);
	void test_2(bool is_asym, bool sign) throw(libtest::test_exception);
	void test_3(bool is_asym, bool sign) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_PERM_TEST_H

