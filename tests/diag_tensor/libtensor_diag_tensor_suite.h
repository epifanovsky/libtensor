#ifndef LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H

#include <libtest/test_suite.h>

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_diag_tensor_tests Tests of diagonal tensor routines
    \brief Unit tests for diagonal tensor classes in libtensor
    \ingroup libtensor_tests
 **/


/** \brief Test suite for diagonal tensor classes and operations in libtensor

    This suite runs the following tests:
     - libtensor::contraction2_test

    \ingroup libtensor_diag_tensor_tests
 **/
class libtensor_diag_tensor_suite : public libtest::test_suite {
private:
//    unit_test_factory<contraction2_test> m_utf_contraction2;

public:
    //! Creates the suite
    libtensor_diag_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_DIAG_TENSOR_SUITE_H

