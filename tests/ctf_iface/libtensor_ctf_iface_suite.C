#include "libtensor_ctf_iface_suite.h"

namespace libtensor {


libtensor_ctf_iface_suite::libtensor_ctf_iface_suite() :
    libtest::test_suite("libtensor_ctf_iface") {

    add_test("ctf_btensor", m_utf_ctf_btensor);
}


} // namespace libtensor
