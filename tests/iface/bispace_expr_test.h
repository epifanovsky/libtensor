#ifndef LIBTENSOR_BISPACE_EXPR_TEST_H
#define    LIBTENSOR_BISPACE_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::bispace_expr::expr_rhs class

    \ingroup libtensor_tests_iface
 **/
class bispace_expr_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_sym_1();
    void test_sym_2();
    void test_sym_3();
    void test_sym_4();
    void test_sym_5();
    void test_sym_6();
    void test_sym_7();
    void test_sym_8();
    void test_sym_9();
    void test_sym_10();

    void test_contains_1();
    void test_contains_2();
    void test_contains_3();
    void test_contains_4();

    void test_locate_1();
    void test_locate_2();
    void test_locate_3();
    void test_locate_4();

    void test_perm_1();
    void test_perm_2();
    void test_perm_3();
    void test_perm_4();
    void test_perm_5();
    void test_perm_6();

    void test_exc_1();

};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_TEST_H

