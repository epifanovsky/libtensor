#include "libtensor_diag_tensor_suite.h"

namespace libtensor {


libtensor_diag_tensor_suite::libtensor_diag_tensor_suite() :
    libtest::test_suite("libtensor_diag_tensor") {

    add_test("diag_tensor", m_utf_diag_tensor);
    add_test("diag_tensor_space", m_utf_diag_tensor_space);
    add_test("diag_tensor_subspace", m_utf_diag_tensor_subspace);
}


}

