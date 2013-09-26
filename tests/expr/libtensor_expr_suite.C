#include "libtensor_expr_suite.h"

namespace libtensor {


libtensor_expr_suite::libtensor_expr_suite() :
    libtest::test_suite("libtensor_expr") {

    add_test("eval_plan", m_utf_eval_plan);
}


} // namespace libtensor
