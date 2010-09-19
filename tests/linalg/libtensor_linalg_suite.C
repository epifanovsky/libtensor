#include <libtensor/libtensor.h>
#include "libtensor_linalg_suite.h"

namespace libtensor {

libtensor_linalg_suite::libtensor_linalg_suite() :
	libtest::test_suite("libtensor_linalg") {

//	add_test("linalg", m_utf_linalg);

	add_test("linalg_x_p_p", m_utf_linalg_x_p_p);
	add_test("linalg_i_i_x", m_utf_linalg_i_i_x);

	add_test("linalg_x_pq_qp", m_utf_linalg_x_pq_qp);
	add_test("linalg_i_ip_p_x", m_utf_linalg_i_ip_p_x);
	add_test("linalg_i_pi_p_x", m_utf_linalg_i_pi_p_x);
	add_test("linalg_ij_i_j_x", m_utf_linalg_ij_i_j_x);

	add_test("linalg_i_ipq_qp_x", m_utf_linalg_i_ipq_qp_x);
	add_test("linalg_ij_ip_jp_x", m_utf_linalg_ij_ip_jp_x);
	add_test("linalg_ij_ip_pj_x", m_utf_linalg_ij_ip_pj_x);
	add_test("linalg_ij_pi_jp_x", m_utf_linalg_ij_pi_jp_x);
	add_test("linalg_ij_pi_pj_x", m_utf_linalg_ij_pi_pj_x);

	add_test("linalg_ij_ipq_jqp_x", m_utf_linalg_ij_ipq_jqp_x);

	add_test("linalg_ijkl_ipl_kpj_x", m_utf_linalg_ijkl_ipl_kpj_x);
}


} // namespace libtensor
