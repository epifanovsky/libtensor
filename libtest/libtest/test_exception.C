#include "test_exception.h"

#include <cstdio>

namespace libtest {


test_exception::test_exception(const char *where, const char *src,
    unsigned lineno, const char *what) {

    snprintf(m_what, 1024, "[%s (%s, %u)] %s", where, src, lineno, what);
}


test_exception::~test_exception() throw() {

}


} // namespace libtest

