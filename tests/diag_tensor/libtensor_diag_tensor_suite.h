#ifndef LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "diag_tensor_test.h"
#include "diag_tensor_space_test.h"
#include "diag_tensor_subspace_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_diag_tensor_tests Tests of diagonal tensor routines
    \brief Unit tests for diagonal tensor classes in libtensor
    \ingroup libtensor_tests
 **/


/** \brief Test suite for diagonal tensor classes and operations in libtensor

    This suite runs the following tests:
     - libtensor::diag_tensor_test
     - libtensor::diag_tensor_space_test
     - libtensor::diag_tensor_subspace_test

    \ingroup libtensor_diag_tensor_tests
 **/
class libtensor_diag_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<diag_tensor_test> m_utf_diag_tensor;
    unit_test_factory<diag_tensor_space_test> m_utf_diag_tensor_space;
    unit_test_factory<diag_tensor_subspace_test> m_utf_diag_tensor_subspace;

public:
    //! Creates the suite
    libtensor_diag_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H

