#include "libtest_suite.h"

namespace libtest {


libtest_suite::libtest_suite() : test_suite("libtest") {

    add_test("test_suite", m_utf_test_suite);
}


} // namespace libtest

