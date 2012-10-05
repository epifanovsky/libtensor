#include "libtensor_diag_block_tensor_suite.h"

namespace libtensor {


libtensor_diag_block_tensor_suite::libtensor_diag_block_tensor_suite() :
    libtest::test_suite("libtensor_diag_block_tensor") {

    add_test("diag_block_tensor", m_utf_diag_block_tensor);
    add_test("diag_btod_random", m_utf_diag_btod_random);
}


} // namespace libtensor

