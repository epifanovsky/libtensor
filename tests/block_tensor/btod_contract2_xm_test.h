#ifndef LIBTENSOR_BTOD_CONTRACT2_XM_TEST_H
#define LIBTENSOR_BTOD_CONTRACT2_XM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_contract2_xm class

    \ingroup libtensor_tests_btod
**/
class btod_contract2_xm_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_bis_1() throw(libtest::test_exception);
    void test_bis_2() throw(libtest::test_exception);
    void test_bis_3() throw(libtest::test_exception);
    void test_bis_4() throw(libtest::test_exception);
    void test_bis_5() throw(libtest::test_exception);

    void test_zeroblk_1() throw(libtest::test_exception);
    void test_zeroblk_2() throw(libtest::test_exception);
    void test_zeroblk_3() throw(libtest::test_exception);
    void test_zeroblk_4() throw(libtest::test_exception);
    void test_zeroblk_5() throw(libtest::test_exception);
    void test_zeroblk_6() throw(libtest::test_exception);

    void test_mat_1() throw(libtest::test_exception);
    void test_mat_2() throw(libtest::test_exception);
    void test_mat_3() throw(libtest::test_exception);

    void test_contr_1() throw(libtest::test_exception);
    void test_contr_1a() throw(libtest::test_exception);
    void test_contr_1b() throw(libtest::test_exception);
    void test_contr_2() throw(libtest::test_exception);
    void test_contr_3() throw(libtest::test_exception);
    void test_contr_4() throw(libtest::test_exception);
    void test_contr_5() throw(libtest::test_exception);
    void test_contr_6() throw(libtest::test_exception);
    void test_contr_7() throw(libtest::test_exception);
    void test_contr_8() throw(libtest::test_exception);
    void test_contr_9() throw(libtest::test_exception);
    void test_contr_10() throw(libtest::test_exception);
    void test_contr_11() throw(libtest::test_exception);
    void test_contr_12() throw(libtest::test_exception);
    void test_contr_13() throw(libtest::test_exception);
    void test_contr_14(double c) throw(libtest::test_exception);
    void test_contr_15(double c) throw(libtest::test_exception);
    void test_contr_16(double c) throw(libtest::test_exception);
    void test_contr_17(double c) throw(libtest::test_exception);
    void test_contr_18(double c) throw(libtest::test_exception);
    void test_contr_19() throw(libtest::test_exception);
    void test_contr_20a() throw(libtest::test_exception);
    void test_contr_20b() throw(libtest::test_exception);
    void test_contr_20c() throw(libtest::test_exception);
    void test_contr_21() throw(libtest::test_exception);
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
