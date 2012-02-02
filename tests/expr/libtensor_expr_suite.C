#include "libtensor_expr_suite.h"

namespace libtensor {


libtensor_expr_suite::libtensor_expr_suite() :
    libtest::test_suite("libtensor") {

    add_test("anytensor", m_utf_anytensor);
    add_test("bispace", m_utf_bispace);
    add_test("bispace_expr", m_utf_bispace_expr);
    add_test("btensor", m_utf_btensor);
    add_test("contract", m_utf_contract);
    add_test("diag", m_utf_diag);
    add_test("dirprod", m_utf_dirprod);
    add_test("dirsum", m_utf_dirsum);
    add_test("labeled_anytensor", m_utf_labeled_anytensor);
    add_test("letter", m_utf_letter);
    add_test("letter_expr", m_utf_letter_expr);
}


} // namespace libtensor
