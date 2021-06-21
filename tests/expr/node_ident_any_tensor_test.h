#ifndef LIBTENSOR_NODE_IDENT_TEST_H
#define LIBTENSOR_NODE_IDENT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::expr::node_ident_any_tensor class

    \ingroup libtensor_tests_iface
**/
class node_ident_any_tensor_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_LIST_TEST_H

