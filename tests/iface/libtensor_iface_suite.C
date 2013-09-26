#include "libtensor_iface_suite.h"

namespace libtensor {

libtensor_iface_suite::libtensor_iface_suite() :
    libtest::test_suite("libtensor") {

    add_test("anon_eval", m_utf_anon_eval);
    add_test("any_tensor", m_utf_any_tensor);
    add_test("bispace", m_utf_bispace);
    add_test("bispace_expr", m_utf_bispace_expr);
    add_test("btensor", m_utf_btensor);
    add_test("contract", m_utf_contract);
    add_test("diag", m_utf_diag);
    add_test("direct_btensor", m_utf_direct_btensor);
    add_test("direct_eval", m_utf_direct_eval);
    add_test("direct_product", m_utf_direct_product);
    add_test("dirsum", m_utf_dirsum);
    add_test("dot_product", m_utf_dot_product);
    add_test("ewmult", m_utf_ewmult);
    add_test("expr", m_utf_expr);
    add_test("expr_tensor", m_utf_expr_tensor);
//    add_test("labeled_btensor", m_utf_labeled_btensor);
    add_test("letter", m_utf_letter);
    add_test("letter_expr", m_utf_letter_expr);
    add_test("mult", m_utf_mult);
    add_test("symm", m_utf_symm);
    add_test("trace", m_utf_trace);
}

} // namespace libtensor
