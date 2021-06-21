#ifndef LIBTENSOR_EXPR_TEST_H
#define LIBTENSOR_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests various problematic expressions

    \ingroup libtensor_tests_iface
 **/
class expr_test : public libtest::unit_test {
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
    void test_12();
    void test_13();
    void test_14();

};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_TEST_H
