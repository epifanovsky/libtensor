#include "libtensor_expr_suite.h"

namespace libtensor {


libtensor_expr_suite::libtensor_expr_suite() :
    libtest::test_suite("libtensor_expr") {

    add_test("eval_plan", m_utf_eval_plan);
    add_test("node_add", m_utf_node_add);
    add_test("node_contract", m_utf_node_contract);
    add_test("node_diag", m_utf_node_diag);
    add_test("node_dirprod", m_utf_node_dirprod);
    add_test("node_dirsum", m_utf_node_dirsum);
    add_test("node_ewmult", m_utf_node_ewmult);
    add_test("node_mult", m_utf_node_mult);
    add_test("node_transform", m_utf_node_transform);
}


} // namespace libtensor
