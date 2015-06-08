#include "libtensor_ctf_iface_suite.h"

namespace libtensor {


libtensor_ctf_iface_suite::libtensor_ctf_iface_suite() :
    libtest::test_suite("libtensor_ctf_iface") {

    add_test("ctf_btensor", m_utf_ctf_btensor);
    add_test("ctf_dot_product", m_utf_ctf_dot_product);
    add_test("ctf_expr", m_utf_ctf_expr);
    add_test("ctf_set", m_utf_ctf_set);
    add_test("ctf_trace", m_utf_ctf_trace);
}


} // namespace libtensor
