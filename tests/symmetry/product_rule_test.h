#ifndef LIBTENSOR_PRODUCT_RULE_TEST_H
#define LIBTENSOR_PRODUCT_RULE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::product_rule class

    \ingroup libtensor_tests_sym
 **/
class product_rule_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
};

} // namespace libtensor

#endif // LIBTENSOR_PRODUCT_RULE_TEST_H

