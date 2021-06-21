#ifndef LIBTENSOR_DIRECT_PRODUCT_TEST_H
#define LIBTENSOR_DIRECT_PRODUCT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the direct product expression

    \ingroup libtensor_tests_iface
**/
class direct_product_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_label_1();
    void test_tt_1();
    void test_tt_2();
    void test_te_1();
    void test_et_1();
    void test_ee_1();
    void test_ee_2();

};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_PRODUCT_TEST_H
