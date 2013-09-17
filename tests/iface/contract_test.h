#ifndef LIBTENSOR_CONTRACT_TEST_H
#define    LIBTENSOR_CONTRACT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::contract function

    \ingroup libtensor_tests_iface
**/
class contract_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_subexpr_labels_1();

    void test_contr_bld_1();
    void test_contr_bld_2();

    void test_tt_1();
    void test_tt_2();
    void test_tt_3();
    void test_tt_4();
    void test_tt_5();
    void test_tt_6();
    void test_tt_7();
    void test_tt_8();
    void test_tt_9();
    void test_te_1();
    void test_te_2();
    void test_te_3();
    void test_te_4();
    void test_et_1();
    void test_et_2();
    void test_et_3();
    void test_ee_1();
    void test_ee_2();
    void test_ee_3();

    void test_contract3_ttt_1();

};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_TEST_H
