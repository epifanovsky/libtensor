#ifndef LIBTENSOR_BTOD_COMPARE_TEST_H
#define LIBTENSOR_BTOD_COMPARE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_compare class

    \ingroup libtensor_tests_btod
**/
class btod_compare_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2a();
    void test_2b();
    void test_3a();
    void test_3b();
    void test_4a();
    void test_4b();
    void test_5a();
    void test_5b();
    void test_6();
    void test_7();

    /** \brief Tests if an exception is throws when the tensors have
            different dimensions
     **/
    void test_exc();

    /** \brief Tests the operation
    **/
    void test_operation();
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_COMPARE_TEST_H

