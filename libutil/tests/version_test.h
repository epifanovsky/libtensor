#ifndef LIBUTIL_VERSION_TEST_H
#define LIBUTIL_VERSION_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::version class

    \ingroup libutil_tests
**/
class version_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_VERSION_TEST_H
