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
    void test_subexpr_labels_1() throw(libtest::test_exception);

    void test_contr_bld_1() throw(libtest::test_exception);
    void test_contr_bld_2() throw(libtest::test_exception);

    void test_tt_1() throw(libtest::test_exception);
    void test_tt_2() throw(libtest::test_exception);
    void test_tt_3() throw(libtest::test_exception);
    void test_tt_4() throw(libtest::test_exception);
    void test_tt_5() throw(libtest::test_exception);
    void test_tt_6() throw(libtest::test_exception);
    void test_tt_7() throw(libtest::test_exception);
    void test_tt_8() throw(libtest::test_exception);
    void test_te_1() throw(libtest::test_exception);
    void test_te_2() throw(libtest::test_exception);
    void test_te_3() throw(libtest::test_exception);
    void test_te_4() throw(libtest::test_exception);
    void test_et_1() throw(libtest::test_exception);
    void test_et_2() throw(libtest::test_exception);
    void test_et_3() throw(libtest::test_exception);
    void test_ee_1() throw(libtest::test_exception);
    void test_ee_2() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT_TEST_H
