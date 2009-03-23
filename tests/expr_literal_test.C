#include "expr_literal_test.h"

namespace libtensor {

void expr_literal_test::perform() throw(libtest::test_exception) {
	expr_literal<double> dexpr1(2.0);
	double d = 3.5;
	expr_literal<double> dexpr2(d);
	expr_literal<double> dexpr3(dexpr1);
}

} // namespace libtensor

