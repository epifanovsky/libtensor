#ifndef LIBTENSOR_LABELED_BTENSOR_TEST_H
#define	LIBTENSOR_LABELED_BTENSOR_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::labeled_btensor class

	\ingroup libtensor_tests
**/
class labeled_btensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_label() throw(libtest::test_exception);
	void test_expr() throw(libtest::test_exception);

	//!	\brief b(i|j) = a(i|j)
	void test_expr_copy_1() throw(libtest::test_exception);

	//!	\brief b(i|j) = a(j|i)
	void test_expr_copy_2() throw(libtest::test_exception);

	//!	\brief b(i|j) = 1.5*a(j|i)
	void test_expr_copy_3() throw(libtest::test_exception);

	//!	\brief b(i|j) = -a(i|j)
	void test_expr_copy_4() throw(libtest::test_exception);

	template<size_t N>
	void compare_ref(const char *test, btensor_i<N, double> &bt,
		btensor_i<N, double> &bt_ref, double thresh)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_TEST_H

