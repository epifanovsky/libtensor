#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::direct_block_tensor class

    \ingroup libtensor_tests_core
**/
class direct_block_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_op_1();
    void test_op_2();
    void test_op_3();
    void test_op_4();
    void test_op_5();
    void test_op_6();

};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_TEST_H

