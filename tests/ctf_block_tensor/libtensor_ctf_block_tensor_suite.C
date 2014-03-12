#include "libtensor_ctf_block_tensor_suite.h"

namespace libtensor {


libtensor_ctf_block_tensor_suite::libtensor_ctf_block_tensor_suite() :
    libtest::test_suite("libtensor_ctf_block_tensor") {

    add_test("ctf_btod_collect", m_utf_ctf_btod_collect);
    add_test("ctf_btod_copy", m_utf_ctf_btod_copy);
    add_test("ctf_btod_set", m_utf_ctf_btod_set);
}


} // namespace libtensor
