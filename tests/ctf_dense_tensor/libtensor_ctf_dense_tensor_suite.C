#include "libtensor_ctf_dense_tensor_suite.h"

namespace libtensor {


libtensor_ctf_dense_tensor_suite::libtensor_ctf_dense_tensor_suite() :
    libtest::test_suite("libtensor_ctf_dense_tensor") {

    add_test("ctf_dense_tensor", m_utf_ctf_dense_tensor);
    add_test("ctf_tod_contract2", m_utf_ctf_tod_contract2);
    add_test("ctf_tod_copy", m_utf_ctf_tod_copy);
    add_test("ctf_tod_diag", m_utf_ctf_tod_diag);
    add_test("ctf_tod_dirsum", m_utf_ctf_tod_dirsum);
    add_test("ctf_tod_distribute", m_utf_ctf_tod_distribute);
    add_test("ctf_tod_dotprod", m_utf_ctf_tod_dotprod);
}


} // namespace libtensor
