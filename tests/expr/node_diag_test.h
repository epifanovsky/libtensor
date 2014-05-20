#ifndef LIBTENSOR_NODE_DIAG_TEST_H
#define LIBTENSOR_NODE_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_diag class

    \ingroup libtensor_tests_expr
**/
class node_diag_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_NODE_DIAG_TEST_H

