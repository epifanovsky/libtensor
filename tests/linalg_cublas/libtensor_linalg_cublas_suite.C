#include "libtensor_linalg_cublas_suite.h"

namespace libtensor {

libtensor_linalg_cublas_suite::libtensor_linalg_cublas_suite() :
    libtest::test_suite("libtensor_linalg_cublas") {

    add_test("linalg_cublas_add1_ij_ij_x", m_utf_linalg_cublas_add1_ij_ij_x);
    add_test("linalg_cublas_add1_ij_ji_x", m_utf_linalg_cublas_add1_ij_ji_x);

    add_test("linalg_cublas_copy_ij_ij_x", m_utf_linalg_cublas_copy_ij_ij_x);
    add_test("linalg_cublas_copy_ij_ji_x", m_utf_linalg_cublas_copy_ij_ji_x);

    add_test("linalg_cublas_mul1_i_x", m_utf_linalg_cublas_mul1_i_x);

    add_test("linalg_cublas_mul2_x_p_p", m_utf_linalg_cublas_mul2_x_p_p);
    add_test("linalg_cublas_mul2_i_i_x", m_utf_linalg_cublas_mul2_i_i_x);
    add_test("linalg_cublas_mul2_i_ip_p_x", m_utf_linalg_cublas_mul2_i_ip_p_x);
    add_test("linalg_cublas_mul2_i_pi_p_x", m_utf_linalg_cublas_mul2_i_pi_p_x);
    add_test("linalg_cublas_mul2_ij_i_j_x", m_utf_linalg_cublas_mul2_ij_i_j_x);
    add_test("linalg_cublas_mul2_ij_ip_jp_x",
        m_utf_linalg_cublas_mul2_ij_ip_jp_x);
    add_test("linalg_cublas_mul2_ij_ip_pj_x",
        m_utf_linalg_cublas_mul2_ij_ip_pj_x);
    add_test("linalg_cublas_mul2_ij_pi_jp_x",
        m_utf_linalg_cublas_mul2_ij_pi_jp_x);
    add_test("linalg_cublas_mul2_ij_pi_pj_x",
        m_utf_linalg_cublas_mul2_ij_pi_pj_x);
}


} // namespace libtensor
