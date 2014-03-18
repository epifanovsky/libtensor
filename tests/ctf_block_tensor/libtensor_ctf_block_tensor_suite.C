#include "libtensor_ctf_block_tensor_suite.h"

namespace libtensor {


libtensor_ctf_block_tensor_suite::libtensor_ctf_block_tensor_suite() :
    libtest::test_suite("libtensor_ctf_block_tensor") {

    add_test("ctf_btod_collect", m_utf_ctf_btod_collect);
    add_test("ctf_btod_contract2", m_utf_ctf_btod_contract2);
    add_test("ctf_btod_copy", m_utf_ctf_btod_copy);
    add_test("ctf_btod_diag", m_utf_ctf_btod_diag);
    add_test("ctf_btod_dirsum", m_utf_ctf_btod_dirsum);
    add_test("ctf_btod_dotprod", m_utf_ctf_btod_dotprod);
    add_test("ctf_btod_ewmult2", m_utf_ctf_btod_ewmult2);
    add_test("ctf_btod_mult", m_utf_ctf_btod_mult);
    add_test("ctf_btod_set", m_utf_ctf_btod_set);
}


} // namespace libtensor
