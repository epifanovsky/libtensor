#ifndef LIBUTIL_COND_TEST_H
#define LIBUTIL_COND_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::cond class

    \ingroup libutil_tests
 **/
class cond_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
    void test_4() throw(libtest::test_exception);
    void test_5() throw(libtest::test_exception);
    void test_6() throw(libtest::test_exception);
    void test_7() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_COND_TEST_H
