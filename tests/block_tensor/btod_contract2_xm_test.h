#ifndef LIBTENSOR_BTOD_CONTRACT2_XM_TEST_H
#define LIBTENSOR_BTOD_CONTRACT2_XM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_contract2_xm class

    \ingroup libtensor_tests_btod
**/
class btod_contract2_xm_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_bis_1();
    void test_bis_2();
    void test_bis_3();
    void test_bis_4();
    void test_bis_5();

    void test_zeroblk_1();
    void test_zeroblk_2();
    void test_zeroblk_3();
    void test_zeroblk_4();
    void test_zeroblk_5();
    void test_zeroblk_6();

    void test_mat_1();
    void test_mat_2();
    void test_mat_3();

    void test_contr_1();
    void test_contr_1a();
    void test_contr_1b();
    void test_contr_2();
    void test_contr_3();
    void test_contr_4();
    void test_contr_5();
    void test_contr_6();
    void test_contr_7();
    void test_contr_8();
    void test_contr_9();
    void test_contr_10();
    void test_contr_11();
    void test_contr_12();
    void test_contr_13();
    void test_contr_14(double c);
    void test_contr_15(double c);
    void test_contr_16(double c);
    void test_contr_17(double c);
    void test_contr_18(double c);
    void test_contr_19();
    void test_contr_20a();
    void test_contr_20b();
    void test_contr_20c();
    void test_contr_21();
    void test_contr_22();
    void test_contr_23();
    void test_contr_24();
    void test_contr_25();
    void test_contr_26();
    void test_contr_27();
    void test_contr_28();

    void test_self_1();
    void test_self_2();
    void test_self_3();

    void test_batch_1();
    void test_batch_2();
    void test_batch_3();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_XM_TEST_H
