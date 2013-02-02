#ifndef LIBTENSOR_LIBTENSOR_CUDA_BLOCK_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_CUDA_BLOCK_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "cuda_btod_contract2_test.h"
#include "cuda_btod_copy_test.h"
#include "cuda_btod_sum_test.h"
#include "cuda_btod_copy_hd_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_cuda_block_tensor_tests Tests of CUDA tensor operations
    \ingroup libtensor_tests
 **/


/** \brief Test suite for CUDA tensor operations in libtensor

    This suite runs the following tests:
     - libtensor::cuda_btod_contract2_test
     - libtensor::cuda_btod_copy_test
     - libtensor::cuda_btod_sum_test
     - libtensor::cuda_btod_copy_hd_test

    \ingroup libtensor_cuda_dense_tensor_tests
 **/
class libtensor_cuda_block_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<cuda_btod_contract2_test> m_utf_cuda_btod_contract2;
    unit_test_factory<cuda_btod_copy_test> m_utf_cuda_btod_copy;
    unit_test_factory<cuda_btod_sum_test> m_utf_cuda_btod_sum;
    unit_test_factory<cuda_btod_copy_hd_test> m_utf_cuda_btod_copy_hd;

public:
    //! Creates the suite
    libtensor_cuda_block_tensor_suite();
};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CUDA_BLOCK_TENSOR_SUITE_H

