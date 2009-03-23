#include "expr_identity.h"
#include "expr_literal.h"
#include "expr_binary_test.h"

namespace libtensor {

typedef expr_literal<int> lit;

void expr_binary_test::perform() throw(libtest::test_exception) {
	lit expr1(1), expr2(2);
	expr_binary< lit, lit, add_op<lit,lit> > expr_bin1(expr1, expr2);
	expr_binary< lit, lit, add_op<lit,lit> > expr_bin2(expr_bin1);
}

} // namespace libtensor

