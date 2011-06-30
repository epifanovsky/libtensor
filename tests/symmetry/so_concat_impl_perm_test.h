#ifndef LIBTENSOR_SO_CONCAT_IMPL_PERM_TEST_H
#define LIBTENSOR_SO_CONCAT_IMPL_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::so_concat_impl_perm class

	\ingroup libtensor_tests_sym
 **/
class so_concat_impl_perm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2(bool symm) throw(libtest::test_exception);
	void test_3(bool symm) throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6() throw(libtest::test_exception);
    void test_7(bool s1, bool s2, bool dirsum) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_CONCAT_IMPL_PERM_TEST_H

