#include "libtensor_linalg_suite.h"

namespace libtensor {

libtensor_linalg_suite::libtensor_linalg_suite() :
    libtest::test_suite("libtensor_linalg") {

    add_test("linalg_add_i_i_x_x", m_utf_linalg_add_i_i_x_x);

    add_test("linalg_copy_ij_ji", m_utf_linalg_copy_ij_ji);

    add_test("linalg_mul2_x_p_p", m_utf_linalg_mul2_x_p_p);
    add_test("linalg_mul2_i_i_i_x", m_utf_linalg_mul2_i_i_i_x);
    add_test("linalg_mul2_i_i_x", m_utf_linalg_mul2_i_i_x);

    add_test("linalg_mul2_x_pq_pq", m_utf_linalg_mul2_x_pq_pq);
    add_test("linalg_mul2_x_pq_qp", m_utf_linalg_mul2_x_pq_qp);
    add_test("linalg_mul2_i_ip_p_x", m_utf_linalg_mul2_i_ip_p_x);
    add_test("linalg_mul2_i_pi_p_x", m_utf_linalg_mul2_i_pi_p_x);
    add_test("linalg_mul2_ij_i_j_x", m_utf_linalg_mul2_ij_i_j_x);

    add_test("linalg_mul2_i_ipq_qp_x", m_utf_linalg_mul2_i_ipq_qp_x);
    add_test("linalg_mul2_ij_ip_jp_x", m_utf_linalg_mul2_ij_ip_jp_x);
    add_test("linalg_mul2_ij_ip_pj_x", m_utf_linalg_mul2_ij_ip_pj_x);
    add_test("linalg_mul2_ij_pi_jp_x", m_utf_linalg_mul2_ij_pi_jp_x);
    add_test("linalg_mul2_ij_pi_pj_x", m_utf_linalg_mul2_ij_pi_pj_x);
    add_test("linalg_blas_version", m_utf_linalg_blas_version);
}


} // namespace libtensor
