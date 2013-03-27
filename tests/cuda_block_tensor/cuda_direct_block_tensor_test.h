#ifndef LIBTENSOR_CUDA_DIRECT_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_CUDA_DIRECT_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/allocator.h>
#include <libtensor/cuda/cuda_allocator.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/cuda_block_tensor/cuda_block_tensor.h>


namespace libtensor {


/** \brief Tests the libtensor::cuda_direct_block_tensor class

    \ingroup libtensor_tests_core
**/
class cuda_direct_block_tensor_test : public libtest::unit_test {

    typedef std_allocator<double> allocator_type;
    typedef cuda_allocator<double> cuda_allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;
    typedef cuda_block_tensor_i_traits<double> cuda_bti_traits;

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

#endif // LIBTENSOR_CUDA_DIRECT_BLOCK_TENSOR_TEST_H

