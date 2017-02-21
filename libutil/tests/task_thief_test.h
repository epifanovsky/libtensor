#ifndef LIBUTIL_TASK_THIEF_TEST_H
#define LIBUTIL_TASK_THIEF_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::task_thief class

     \ingroup libutil_tests
 **/
class task_thief_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libutil

#endif // LIBUTIL_TASK_THIEF_TEST_H
