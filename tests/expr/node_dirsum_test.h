#ifndef LIBTENSOR_NODE_DIRSUM_TEST_H
#define LIBTENSOR_NODE_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::node_dirsum class

    \ingroup libtensor_tests_expr
**/
class node_dirsum_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_NODE_DIRSUM_TEST_H

