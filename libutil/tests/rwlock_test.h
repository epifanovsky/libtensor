#ifndef LIBUTIL_RWLOCK_TEST_H
#define LIBUTIL_RWLOCK_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::rwlock class

    \ingroup libutil_tests
 **/
class rwlock_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a() throw(libtest::test_exception);
    void test_1b() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);

};


} // namespace libutil

#endif // LIBUTIL_RWLOCK_TEST_H
