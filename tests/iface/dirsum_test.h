#ifndef LIBTENSOR_DIRSUM_TEST_H
#define    LIBTENSOR_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::dirsum function

    \ingroup libtensor_tests_iface
**/
class dirsum_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_tt_1();
    void test_tt_2();
    void test_tt_3();
    void test_tt_4();
    void test_tt_5();
    void test_tt_6();
    void test_te_1();
    void test_et_1();
    void test_ee_1();

};

} // namespace libtensor

#endif // LIBTENSOR_DIRSUM_TEST_H
