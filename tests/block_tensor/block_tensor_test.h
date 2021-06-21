#ifndef LIBTENSOR_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::block_tensor class

    \ingroup libtensor_tests_core
 **/
class block_tensor_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_nonzero_blocks_1();
    void test_nonzero_blocks_2();

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_TEST_H
