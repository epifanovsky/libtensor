#ifndef LIBTENSOR_MULT_TEST_H
#define    LIBTENSOR_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::mult function

    \ingroup libtensor_tests_iface
**/
class mult_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_tt_1a();
    void test_tt_1b();
    void test_tt_2();
    void test_tt_3();
    void test_tt_4();
    void test_tt_5();
    void test_tt_6a();
    void test_tt_6b();
    void test_te_1();
    void test_te_2();
    void test_te_3();
    void test_et_1();
    void test_et_2();
    void test_et_3();
    void test_ee_1a();
    void test_ee_1b();
    void test_ee_2();

};

} // namespace libtensor

#endif // LIBTENSOR_MULT_TEST_H
