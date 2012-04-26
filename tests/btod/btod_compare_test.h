#ifndef LIBTENSOR_BTOD_COMPARE_TEST_H
#define LIBTENSOR_BTOD_COMPARE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_compare class

    \ingroup libtensor_tests_btod
**/
class btod_compare_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2a() throw(libtest::test_exception);
    void test_2b() throw(libtest::test_exception);
    void test_3a() throw(libtest::test_exception);
    void test_3b() throw(libtest::test_exception);
    void test_4a() throw(libtest::test_exception);
    void test_4b() throw(libtest::test_exception);
    void test_5a() throw(libtest::test_exception);
    void test_5b() throw(libtest::test_exception);
    void test_6() throw(libtest::test_exception);

    /** \brief Tests if an exception is throws when the tensors have
            different dimensions
     **/
    void test_exc() throw(libtest::test_exception);

    /** \brief Tests the operation
    **/
    void test_operation() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_COMPARE_TEST_H

