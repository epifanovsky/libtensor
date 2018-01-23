#ifndef LIBUTIL_THREAD_TEST_H
#define LIBUTIL_THREAD_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::thread class

     \ingroup libutil_tests
 **/
class thread_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_THREAD_TEST_H
