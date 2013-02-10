#include "libtensor_ctf_dense_tensor_suite.h"

namespace libtensor {


libtensor_ctf_dense_tensor_suite::libtensor_ctf_dense_tensor_suite() :
    libtest::test_suite("libtensor_ctf_dense_tensor") {

    add_test("ctf_dense_tensor", m_utf_ctf_dense_tensor);
}


} // namespace libtensor
