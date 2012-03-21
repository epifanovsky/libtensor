#include "libtensor_linalg_suite.h"

namespace libtensor {

libtensor_linalg_suite::libtensor_linalg_suite() :
    libtest::test_suite("libtensor_linalg") {

    add_test("linalg_add_i_i_x_x", m_utf_linalg_add_i_i_x_x);

    add_test("linalg_x_p_p", m_utf_linalg_x_p_p);
    add_test("linalg_i_i_i_x", m_utf_linalg_i_i_i_x);
    add_test("linalg_i_i_x", m_utf_linalg_i_i_x);

    add_test("linalg_x_pq_qp", m_utf_linalg_x_pq_qp);
    add_test("linalg_i_ip_p_x", m_utf_linalg_i_ip_p_x);
    add_test("linalg_i_pi_p_x", m_utf_linalg_i_pi_p_x);
    add_test("linalg_ij_i_j_x", m_utf_linalg_ij_i_j_x);
    add_test("linalg_ij_ji", m_utf_linalg_ij_ji);

    add_test("linalg_i_ipq_qp_x", m_utf_linalg_i_ipq_qp_x);
    add_test("linalg_ij_ip_jp_x", m_utf_linalg_ij_ip_jp_x);
    add_test("linalg_ij_ip_pj_x", m_utf_linalg_ij_ip_pj_x);
    add_test("linalg_ij_pi_jp_x", m_utf_linalg_ij_pi_jp_x);
    add_test("linalg_ij_pi_pj_x", m_utf_linalg_ij_pi_pj_x);
    add_test("linalg_ij_pji_p_x", m_utf_linalg_ij_pji_p_x);

    add_test("linalg_ij_ipq_jqp_x", m_utf_linalg_ij_ipq_jqp_x);
    add_test("linalg_ijk_ip_pkj_x", m_utf_linalg_ijk_ip_pkj_x);
    add_test("linalg_ijk_pi_pkj_x", m_utf_linalg_ijk_pi_pkj_x);

    add_test("linalg_ijk_ipq_kjqp_x", m_utf_linalg_ijk_ipq_kjqp_x);
    add_test("linalg_ijk_ipkq_jpq_x", m_utf_linalg_ijk_ipkq_jpq_x);
    add_test("linalg_ijk_pikq_jpq_x", m_utf_linalg_ijk_pikq_jpq_x);
    add_test("linalg_ijk_piqk_jpq_x", m_utf_linalg_ijk_piqk_jpq_x);
    add_test("linalg_ijk_pkiq_pjq_x", m_utf_linalg_ijk_pkiq_pjq_x);
    add_test("linalg_ijkl_ikp_jpl_x", m_utf_linalg_ijkl_ikp_jpl_x);
    add_test("linalg_ijkl_ipj_plk_x", m_utf_linalg_ijkl_ipj_plk_x);
    add_test("linalg_ijkl_ipl_jpk_x", m_utf_linalg_ijkl_ipl_jpk_x);
    add_test("linalg_ijkl_ipl_kpj_x", m_utf_linalg_ijkl_ipl_kpj_x);

    add_test("linalg_ijkl_ipkq_pljq_x", m_utf_linalg_ijkl_ipkq_pljq_x);
    add_test("linalg_ijkl_iplq_kpjq_x", m_utf_linalg_ijkl_iplq_kpjq_x);
    add_test("linalg_ijkl_iplq_pkjq_x", m_utf_linalg_ijkl_iplq_pkjq_x);
    add_test("linalg_ijkl_iplq_pkqj_x", m_utf_linalg_ijkl_iplq_pkqj_x);
    add_test("linalg_ijkl_ipqk_pljq_x", m_utf_linalg_ijkl_ipqk_pljq_x);
    add_test("linalg_ijkl_ipql_pkjq_x", m_utf_linalg_ijkl_ipql_pkjq_x);
    add_test("linalg_ijkl_ipql_pkqj_x", m_utf_linalg_ijkl_ipql_pkqj_x);
    add_test("linalg_ijkl_ipql_qkpj_x", m_utf_linalg_ijkl_ipql_qkpj_x);
    add_test("linalg_ijkl_ipql_qpkj_x", m_utf_linalg_ijkl_ipql_qpkj_x);
    add_test("linalg_ijkl_pikq_pljq_x", m_utf_linalg_ijkl_pikq_pljq_x);
    add_test("linalg_ijkl_pilq_kpjq_x", m_utf_linalg_ijkl_pilq_kpjq_x);
    add_test("linalg_ijkl_pilq_pkjq_x", m_utf_linalg_ijkl_pilq_pkjq_x);
    add_test("linalg_ijkl_piqk_pljq_x", m_utf_linalg_ijkl_piqk_pljq_x);
    add_test("linalg_ijkl_piql_kpqj_x", m_utf_linalg_ijkl_piql_kpqj_x);
    add_test("linalg_ijkl_piql_pkjq_x", m_utf_linalg_ijkl_piql_pkjq_x);
    add_test("linalg_ijkl_piql_pkqj_x", m_utf_linalg_ijkl_piql_pkqj_x);
    add_test("linalg_ijkl_piql_qkpj_x", m_utf_linalg_ijkl_piql_qkpj_x);
    add_test("linalg_ijkl_piql_qpkj_x", m_utf_linalg_ijkl_piql_qpkj_x);
    add_test("linalg_ijkl_pkiq_jplq_x", m_utf_linalg_ijkl_pkiq_jplq_x);
    add_test("linalg_ijkl_pkiq_jpql_x", m_utf_linalg_ijkl_pkiq_jpql_x);
    add_test("linalg_ijkl_pkiq_pjlq_x", m_utf_linalg_ijkl_pkiq_pjlq_x);
    add_test("linalg_ijkl_pkiq_pjql_x", m_utf_linalg_ijkl_pkiq_pjql_x);
    add_test("linalg_ijkl_pliq_jpkq_x", m_utf_linalg_ijkl_pliq_jpkq_x);
    add_test("linalg_ijkl_pliq_jpqk_x", m_utf_linalg_ijkl_pliq_jpqk_x);
    add_test("linalg_ijkl_pliq_pjqk_x", m_utf_linalg_ijkl_pliq_pjqk_x);

    add_test("linalg_ijklm_ipkm_jlp_x", m_utf_linalg_ijklm_ipkm_jlp_x);
}


} // namespace libtensor
