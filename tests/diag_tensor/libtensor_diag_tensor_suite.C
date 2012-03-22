#include "libtensor_diag_tensor_suite.h"

namespace libtensor {


libtensor_diag_tensor_suite::libtensor_diag_tensor_suite() :
    libtest::test_suite("libtensor_diag_tensor") {

    add_test("diag_tensor", m_utf_diag_tensor);
    add_test("diag_tensor_space", m_utf_diag_tensor_space);
    add_test("diag_tensor_subspace", m_utf_diag_tensor_subspace);
    add_test("diag_to_add_space", m_utf_diag_to_add_space);
    add_test("diag_to_contract2_space", m_utf_diag_to_contract2_space);
    add_test("diag_tod_adjust_space", m_utf_diag_tod_adjust_space);
    add_test("diag_tod_set", m_utf_diag_tod_set);
    add_test("tod_conv_diag_tensor", m_utf_tod_conv_diag_tensor);
}


} // namespace libtensor

