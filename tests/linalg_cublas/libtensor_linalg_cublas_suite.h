#ifndef LIBTENSOR_LIBTENSOR_LINALG_CUBLAS_SUITE_H
#define LIBTENSOR_LIBTENSOR_LINALG_CUBLAS_SUITE_H

#include <libtest/test_suite.h>
#include "linalg_cublas_add1_ij_ij_x_test.h"
#include "linalg_cublas_add1_ij_ji_x_test.h"
#include "linalg_cublas_copy_ij_ij_x_test.h"
#include "linalg_cublas_copy_ij_ji_x_test.h"
#include "linalg_cublas_mul1_i_x_test.h"
#include "linalg_cublas_mul2_x_p_p_test.h"
#include "linalg_cublas_mul2_i_i_x_test.h"
#include "linalg_cublas_mul2_i_ip_p_x_test.h"
#include "linalg_cublas_mul2_i_pi_p_x_test.h"
#include "linalg_cublas_mul2_ij_i_j_x_test.h"
#include "linalg_cublas_mul2_ij_ip_jp_x_test.h"
#include "linalg_cublas_mul2_ij_ip_pj_x_test.h"
#include "linalg_cublas_mul2_ij_pi_jp_x_test.h"
#include "linalg_cublas_mul2_ij_pi_pj_x_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_linalg_cublas Tests of CUDA linear algebra
        components
    \brief Unit tests of the CUDA linear algebra components in libtensor
    \ingroup libtensor_tests
 **/


/** \brief Test suite for the CUDA linear algebra in the tensor library
    \ingroup libtensor_tests

    This suite runs the following tests:
     - libtensor::linalg_cublas_add1_ij_ij_x_test
     - libtensor::linalg_cublas_add1_ij_ji_x_test
     - libtensor::linalg_cublas_copy_ij_ij_x_test
     - libtensor::linalg_cublas_copy_ij_ji_x_test
     - libtensor::linalg_cublas_mul1_i_x_test
     - libtensor::linalg_cublas_mul2_x_p_p_test
     - libtensor::linalg_cublas_mul2_i_i_x_test
     - libtensor::linalg_cublas_mul2_i_ip_p_x_test
     - libtensor::linalg_cublas_mul2_i_pi_p_x_test
     - libtensor::linalg_cublas_mul2_ij_i_j_x_test
     - libtensor::linalg_cublas_mul2_ij_ip_jp_x_test
     - libtensor::linalg_cublas_mul2_ij_ip_pj_x_test
     - libtensor::linalg_cublas_mul2_ij_pi_jp_x_test
     - libtensor::linalg_cublas_mul2_ij_pi_pj_x_test

 **/
class libtensor_linalg_cublas_suite : public libtest::test_suite {
private:
    unit_test_factory<linalg_cublas_add1_ij_ij_x_test>
        m_utf_linalg_cublas_add1_ij_ij_x;
    unit_test_factory<linalg_cublas_add1_ij_ji_x_test>
        m_utf_linalg_cublas_add1_ij_ji_x;
    unit_test_factory<linalg_cublas_copy_ij_ij_x_test>
        m_utf_linalg_cublas_copy_ij_ij_x;
    unit_test_factory<linalg_cublas_copy_ij_ji_x_test>
        m_utf_linalg_cublas_copy_ij_ji_x;
    unit_test_factory<linalg_cublas_mul1_i_x_test>
        m_utf_linalg_cublas_mul1_i_x;
    unit_test_factory<linalg_cublas_mul2_x_p_p_test>
        m_utf_linalg_cublas_mul2_x_p_p;
    unit_test_factory<linalg_cublas_mul2_i_i_x_test>
        m_utf_linalg_cublas_mul2_i_i_x;
    unit_test_factory<linalg_cublas_mul2_i_ip_p_x_test>
        m_utf_linalg_cublas_mul2_i_ip_p_x;
    unit_test_factory<linalg_cublas_mul2_i_pi_p_x_test>
        m_utf_linalg_cublas_mul2_i_pi_p_x;
    unit_test_factory<linalg_cublas_mul2_ij_i_j_x_test>
        m_utf_linalg_cublas_mul2_ij_i_j_x;
    unit_test_factory<linalg_cublas_mul2_ij_ip_jp_x_test>
        m_utf_linalg_cublas_mul2_ij_ip_jp_x;
    unit_test_factory<linalg_cublas_mul2_ij_ip_pj_x_test>
        m_utf_linalg_cublas_mul2_ij_ip_pj_x;
    unit_test_factory<linalg_cublas_mul2_ij_pi_jp_x_test> 
        m_utf_linalg_cublas_mul2_ij_pi_jp_x;
    unit_test_factory<linalg_cublas_mul2_ij_pi_pj_x_test>
        m_utf_linalg_cublas_mul2_ij_pi_pj_x;

public:
    //! Creates the suite
    libtensor_linalg_cublas_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_LINALG_CUBLAS_SUITE_H

