#ifndef LIBTENSOR_DOT_PRODUCT_TEST_H
#define    LIBTENSOR_DOT_PRODUCT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::dot_product function

    \ingroup libtensor_tests_iface
**/
class dot_product_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_tt_ij_ij_1();
    void test_tt_ij_ji_1();
    void test_te_ij_ij_1();
    void test_te_ij_ji_1();
    void test_et_1();

    void check_ref(const char *testname, double d, double d_ref)
       ;
};

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_TEST_H
