#include "libtensor_cuda_dense_tensor_suite.h"

namespace libtensor {


libtensor_cuda_dense_tensor_suite::libtensor_cuda_dense_tensor_suite() :
    libtest::test_suite("libtensor_cuda_dense_tensor") {

    add_test("tod_cuda_copy", m_utf_tod_cuda_copy);
    add_test("tod_set_cuda", m_utf_tod_set_cuda);
    add_test("tod_add_cuda", m_utf_tod_add_cuda);
}


} // namespace libtensor

