#ifndef LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tod_conv_diag_block_tensor class

    \ingroup libtensor_diag_block_tensor_tests
**/
class tod_conv_diag_block_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_TEST_H
