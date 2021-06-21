#ifndef LIBTENSOR_BTOD_APPLY_TEST_H
#define LIBTENSOR_BTOD_APPLY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_apply class

    \ingroup libtensor_tests_btod
 **/
class btod_apply_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_zero_1();
    void test_zero_2();
    void test_zero_3();
    void test_nosym_1();
    void test_nosym_2();
    void test_nosym_3();
    void test_nosym_4();
    void test_sym_1();
    void test_sym_2();
    void test_sym_3();
    void test_sym_4();
    void test_sym_5();
    void test_add_nosym_1();
    void test_add_nosym_2();
    void test_add_nosym_3();
    void test_add_nosym_4();
    void test_add_eqsym_1();
    void test_add_eqsym_2();
    void test_add_eqsym_3();
    void test_add_eqsym_4();
    void test_add_eqsym_5();
    void test_add_nesym_1();
    void test_add_nesym_2();
    void test_add_nesym_3();
    void test_add_nesym_4();
    void test_add_nesym_5();
    void test_add_nesym_5_sp();
    void test_add_nesym_6();
    void test_add_nesym_7_sp1();
    void test_add_nesym_7_sp2();
    void test_add_nesym_7_sp3();

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_APPLY_TEST_H
