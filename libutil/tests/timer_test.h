#ifndef LIBUTIL_TIMER_TEST_H
#define LIBUTIL_TIMER_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::timer class

    \ingroup libutil_tests
**/
class timer_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_TIMER_TEST_H
