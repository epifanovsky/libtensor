#ifndef LIBTENSOR_EXPR_TENSOR_TEST_H
#define LIBTENSOR_EXPR_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::expr_tensor class

    \ingroup libtensor_tests_iface
**/
class expr_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();

};


} // namespace libtensor

#endif // LIBTENSOR_EXPR_TENSOR_TEST_H

