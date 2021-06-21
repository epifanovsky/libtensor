#ifndef LIBTENSOR_NODE_SET_TEST_H
#define LIBTENSOR_NODE_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_set class

    \ingroup libtensor_tests_expr
**/
class node_set_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_NODE_SET_TEST_H

