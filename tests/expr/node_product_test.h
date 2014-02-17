#ifndef LIBTENSOR_NODE_PRODUCT_TEST_H
#define LIBTENSOR_NODE_PRODUCT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_product class

    \ingroup libtensor_tests_expr
**/
class node_product_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();

};


} // namespace libtensor


#endif // LIBTENSOR_NODE_PRODUCT_TEST_H
