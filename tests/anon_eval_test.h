#ifndef LIBTENSOR_ANON_EVAL_TEST_H
#define LIBTENSOR_ANON_EVAL_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_tensor_i.h>
#include <libtensor/iface/labeled_btensor_expr.h>
#include <libtensor/iface/letter_expr.h>

namespace libtensor {

/**	\brief Tests the libtensor::labeled_btensor_expr::anon_eval class

	\ingroup libtensor_tests
 **/
class anon_eval_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<size_t N, typename T, typename Core>
	void invoke_eval(
		const char *testname,
		const labeled_btensor_expr::expr<N, T, Core> &expr,
		const letter_expr<N> &label, block_tensor_i<N, T> &ref,
		double thresh = 1e-14)
		throw(libtest::test_exception);

	void test_copy_1() throw(libtest::test_exception);
	void test_copy_2() throw(libtest::test_exception);
	void test_copy_3() throw(libtest::test_exception);
	void test_copy_4() throw(libtest::test_exception);
	void test_copy_5() throw(libtest::test_exception);
	void test_copy_6() throw(libtest::test_exception);

	void test_add_1() throw(libtest::test_exception);

	void test_contr_1() throw(libtest::test_exception);
	void test_contr_2() throw(libtest::test_exception);

	void test_mixed_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_ANON_EVAL_TEST_H
