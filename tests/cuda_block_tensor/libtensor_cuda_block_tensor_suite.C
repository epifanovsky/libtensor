#include "libtensor_cuda_block_tensor_suite.h"

namespace libtensor {


libtensor_cuda_block_tensor_suite::libtensor_cuda_block_tensor_suite() :
    libtest::test_suite("libtensor_cuda_block_tensor") {

//    add_test("cuda_btod_contract2", m_utf_cuda_btod_contract2);
//    add_test("cuda_btod_copy", m_utf_cuda_btod_copy);
    add_test("cuda_tod_bcopy_hd", m_utf_cuda_btod_copy_hd);
}


} // namespace libtensor

