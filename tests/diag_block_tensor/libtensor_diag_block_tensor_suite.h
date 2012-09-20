#ifndef LIBTENSOR_LIBTENSOR_DIAG_BLOCK_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_DIAG_BLOCK_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "diag_block_tensor_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_diag_block_tensor_tests Tests of diagonal block tensor
        routines
    \brief Unit tests for diagonal block tensor classes in libtensor
    \ingroup libtensor_tests
 **/


/** \brief Test suite for diagonal tensor classes and operations in libtensor

    This suite runs the following tests:
     - libtensor::diag_block_tensor_test

    \ingroup libtensor_diag_block_tensor_tests
 **/
class libtensor_diag_block_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<diag_block_tensor_test> m_utf_diag_block_tensor;

public:
    //! Creates the suite
    libtensor_diag_block_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_DIAG_BLOCK_TENSOR_SUITE_H

