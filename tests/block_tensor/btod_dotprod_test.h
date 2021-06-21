#ifndef LIBTENSOR_BTOD_DOTPROD_TEST_H
#define LIBTENSOR_BTOD_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_dotprod class

    \ingroup libtensor_tests_btod
**/
class btod_dotprod_test : public libtest::unit_test {
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
    void test_10a();
    void test_10b();
    void test_10c(bool both);
    void test_11();
    void test_12();
    void test_13a();
    void test_13b();

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_TEST_H
