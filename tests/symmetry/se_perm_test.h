#ifndef LIBTENSOR_SE_PERM_TEST_H
#define LIBTENSOR_SE_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::se_perm class

    \ingroup libtensor_tests_sym
 **/
class se_perm_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_sym_ab_ba();
    void test_asym_ab_ba();
    void test_sym_abc_bca();
    void test_asym_abc_bca();
    void test_sym_abcd_badc();
    void test_asym_abcd_badc();
};

} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_TEST_H

