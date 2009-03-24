#include "expression_test.h"
#include "expr.h"
#include "expr_binary.h"
#include "expr_literal.h"
#include "expr_add_functor.h"

namespace libtensor {

void expression_test::perform() throw(libtest::test_exception) {
	test_1();
}

void expression_test::test_1() throw(libtest::test_exception) {
	typedef expr_literal<double> expr_lit_t;
	typedef expr_add_functor<expr_lit_t,expr_lit_t> expr_add_func_t;
	typedef expr_binary< expr_literal<double>, expr_literal<double>,
		expr_add_func_t > expr_sum_t;

	expr<expr_lit_t> d1(2.0), d2(3.0);
//	expr<expr_sum_t> dsum(d1+d2);
}

} // namespace libtensor

