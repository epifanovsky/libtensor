#ifndef LIBUTIL_TIMINGS_TEST_H
#define LIBUTIL_TIMINGS_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::timings class

    \ingroup libutil_tests
**/
class timings_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_TIMINGS_TEST_H
