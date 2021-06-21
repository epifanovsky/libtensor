#ifndef LIBTENSOR_SYMM_TEST_H
#define    LIBTENSOR_SYMM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::symm function

    \ingroup libtensor_tests_iface
**/
class symm_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_symm2_contr_tt_1();
    void test_symm2_contr_tt_2();
    void test_symm2_contr_ee_1();
    void test_asymm2_contr_tt_1();
    void test_asymm2_contr_tt_2();
    void test_asymm2_contr_tt_3();
    void test_asymm2_contr_tt_4();
    void test_asymm2_contr_tt_5();
    void test_asymm2_contr_tt_6();
    void test_asymm2_contr_et_1();
    void test_asymm2_contr_ee_1();
    void test_asymm2_contr_ee_2();

    void test_symm22_t_1();
    void test_asymm22_t_1();
    void test_symm22_t_2();
    void test_asymm22_t_2();

    void test_symm22_e_1();
    void test_asymm22_e_1();
    void test_symm22_e_2();
    void test_asymm22_e_2();

    void test_symm3_t_1();
    void test_asymm3_e_1();
};

} // namespace libtensor

#endif // LIBTENSOR_SYMM_TEST_H
