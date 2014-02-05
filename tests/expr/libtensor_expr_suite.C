#include "libtensor_expr_suite.h"

namespace libtensor {


libtensor_expr_suite::libtensor_expr_suite() :
    libtest::test_suite("libtensor_expr") {

//    add_test("eval_plan", m_utf_eval_plan);
    add_test("expr_tree", m_utf_expr_tree);
    add_test("graph", m_utf_graph);
    add_test("node_add", m_utf_node_add);
    add_test("node_contract", m_utf_node_contract);
    add_test("node_diag", m_utf_node_diag);
    add_test("node_dot_product", m_utf_node_dot_product);
    add_test("node_ident", m_utf_node_ident);
    add_test("node_product", m_utf_node_product);
    add_test("node_scalar", m_utf_node_scalar);
    add_test("node_trace", m_utf_node_trace);
    add_test("node_transform", m_utf_node_transform);
}


} // namespace libtensor
