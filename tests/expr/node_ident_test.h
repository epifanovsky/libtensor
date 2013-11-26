#ifndef LIBTENSOR_NODE_IDENT_TEST_H
#define LIBTENSOR_NODE_IDENT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_ident class

    \ingroup libtensor_tests_iface
**/
class node_ident_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
};


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_LIST_TEST_H

