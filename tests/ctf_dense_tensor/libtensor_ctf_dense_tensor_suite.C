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
    add_test("ctf_tod_ewmult2", m_utf_ctf_tod_ewmult2);
    add_test("ctf_tod_mult", m_utf_ctf_tod_mult);
    add_test("ctf_tod_scale", m_utf_ctf_tod_scale);
    add_test("ctf_tod_scatter", m_utf_ctf_tod_scatter);
    add_test("ctf_tod_set", m_utf_ctf_tod_set);
    add_test("ctf_tod_set_diag", m_utf_ctf_tod_set_diag);
    add_test("ctf_tod_trace", m_utf_ctf_tod_trace);
}


} // namespace libtensor
