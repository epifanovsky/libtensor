#include "unit_test.h"

namespace libtest {


void unit_test::fail_test(const char *where, const char *src, unsigned lineno,
    const char *what) throw(test_exception) {

    throw test_exception(where, src, lineno, what);
}


} // namespace libtest

