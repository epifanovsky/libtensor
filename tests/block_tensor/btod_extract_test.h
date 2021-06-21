#ifndef LIBTENSOR_BTOD_EXTRACT_TEST_H
#define LIBTENSOR_BTOD_EXTRACT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_extract class

    \ingroup libtensor_tests_btod
**/
class btod_extract_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8();
    void test_9();
    void test_10();
    void test_11();
    void test_12a();
    void test_12b();
    void test_12c();
    void test_13a();
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXTRACT_TEST_H
