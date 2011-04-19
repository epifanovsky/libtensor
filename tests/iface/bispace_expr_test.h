#ifndef LIBTENSOR_BISPACE_EXPR_TEST_H
#define	LIBTENSOR_BISPACE_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::bispace_expr::expr class

	\ingroup libtensor_tests
 **/
class bispace_expr_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_sym_1() throw(libtest::test_exception);
	void test_sym_2() throw(libtest::test_exception);
	void test_sym_3() throw(libtest::test_exception);
	void test_sym_4() throw(libtest::test_exception);
	void test_sym_5() throw(libtest::test_exception);
	void test_sym_6() throw(libtest::test_exception);
	void test_sym_7() throw(libtest::test_exception);
	void test_sym_8() throw(libtest::test_exception);
	void test_sym_9() throw(libtest::test_exception);
	void test_sym_10() throw(libtest::test_exception);

	void test_contains_1() throw(libtest::test_exception);
	void test_contains_2() throw(libtest::test_exception);
	void test_contains_3() throw(libtest::test_exception);
	void test_contains_4() throw(libtest::test_exception);

	void test_locate_1() throw(libtest::test_exception);
	void test_locate_2() throw(libtest::test_exception);
	void test_locate_3() throw(libtest::test_exception);
	void test_locate_4() throw(libtest::test_exception);

	void test_perm_1() throw(libtest::test_exception);
	void test_perm_2() throw(libtest::test_exception);
	void test_perm_3() throw(libtest::test_exception);
	void test_perm_4() throw(libtest::test_exception);
	void test_perm_5() throw(libtest::test_exception);
	void test_perm_6() throw(libtest::test_exception);

	void test_exc_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_TEST_H

