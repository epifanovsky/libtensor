#ifndef LIBUTIL_TLS_TEST_H
#define LIBUTIL_TLS_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::tls class
 **/
class tls_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_TLS_TEST_H
