#ifndef LIBTENSOR_NODE_TRACE_TEST_H
#define LIBTENSOR_NODE_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_trace class

    \ingroup libtensor_tests_expr
**/
class node_trace_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_NODE_TRACE_TEST_H
