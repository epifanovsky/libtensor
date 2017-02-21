#ifndef LIBUTIL_BACKTRACE_TEST_H
#define LIBUTIL_BACKTRACE_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::backtrace class

    \ingroup libutil_tests
**/
class backtrace_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_BACKTRACE_TEST_H
