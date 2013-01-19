#include "libtensor_cuda_dense_tensor_suite.h"

namespace libtensor {


libtensor_cuda_dense_tensor_suite::libtensor_cuda_dense_tensor_suite() :
    libtest::test_suite("libtensor_cuda_dense_tensor") {

    add_test("cuda_tod_contract2", m_utf_cuda_tod_contract2);
    add_test("tod_cuda_copy", m_utf_tod_cuda_copy);
    add_test("cuda_tod_copy_hd", m_utf_cuda_tod_copy_hd);
    add_test("cuda_tod_set", m_utf_cuda_tod_set);
    add_test("tod_add_cuda", m_utf_tod_add_cuda);
}


} // namespace libtensor

