#ifndef LIBTENSOR_NODE_DOT_PRODUCT_TEST_H
#define LIBTENSOR_NODE_DOT_PRODUCT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_dot_product class

    \ingroup libtensor_tests_expr
**/
class node_dot_product_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_NODE_DOT_PRODUCT_TEST_H
