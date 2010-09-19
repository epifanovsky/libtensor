#ifndef LIBTENSOR_LIBTENSOR_LINALG_SUITE_H
#define LIBTENSOR_LIBTENSOR_LINALG_SUITE_H

#include <libtest/test_suite.h>
#include "linalg_test.h"
#include "linalg_x_p_p_test.h"
#include "linalg_i_i_x_test.h"
#include "linalg_x_pq_qp_test.h"
#include "linalg_i_ip_p_x_test.h"
#include "linalg_i_pi_p_x_test.h"
#include "linalg_ij_i_j_x_test.h"
#include "linalg_i_ipq_qp_x_test.h"
#include "linalg_ij_ip_jp_x_test.h"
#include "linalg_ij_ip_pj_x_test.h"
#include "linalg_ij_pi_jp_x_test.h"
#include "linalg_ij_pi_pj_x_test.h"
#include "linalg_ij_ipq_jqp_x_test.h"
#include "linalg_ijkl_ipl_kpj_x_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/**	\brief Test suite for the linear algebra in the tensor library

	This suite runs the following tests:
	 - libtensor::linalg_test
	 - libtensor::linalg_x_p_p_test
	 - libtensor::linalg_i_i_x_test
	 - libtensor::linalg_x_pq_qp_test
	 - libtensor::linalg_i_ip_p_x_test
	 - libtensor::linalg_i_pi_p_x_test
	 - libtensor::linalg_ij_i_j_x_test
	 - libtensor::linalg_i_ipq_qp_x_test
	 - libtensor::linalg_ij_ip_jp_x_test
	 - libtensor::linalg_ij_ip_pj_x_test
	 - libtensor::linalg_ij_pi_jp_x_test
	 - libtensor::linalg_ij_pi_pj_x_test
	 - libtensor::linalg_ij_ipq_jqp_x_test
	 - libtensor::linalg_ijkl_ipl_kpj_x_test

	\ingroup libtensor_tests
 **/
class libtensor_linalg_suite : public libtest::test_suite {
private:
	unit_test_factory<linalg_test> m_utf_linalg;
	unit_test_factory<linalg_x_p_p_test> m_utf_linalg_x_p_p;
	unit_test_factory<linalg_i_i_x_test> m_utf_linalg_i_i_x;
	unit_test_factory<linalg_x_pq_qp_test> m_utf_linalg_x_pq_qp;
	unit_test_factory<linalg_i_ip_p_x_test> m_utf_linalg_i_ip_p_x;
	unit_test_factory<linalg_i_pi_p_x_test> m_utf_linalg_i_pi_p_x;
	unit_test_factory<linalg_ij_i_j_x_test> m_utf_linalg_ij_i_j_x;
	unit_test_factory<linalg_i_ipq_qp_x_test> m_utf_linalg_i_ipq_qp_x;
	unit_test_factory<linalg_ij_ip_jp_x_test> m_utf_linalg_ij_ip_jp_x;
	unit_test_factory<linalg_ij_ip_pj_x_test> m_utf_linalg_ij_ip_pj_x;
	unit_test_factory<linalg_ij_pi_jp_x_test> m_utf_linalg_ij_pi_jp_x;
	unit_test_factory<linalg_ij_pi_pj_x_test> m_utf_linalg_ij_pi_pj_x;
	unit_test_factory<linalg_ij_ipq_jqp_x_test> m_utf_linalg_ij_ipq_jqp_x;
	unit_test_factory<linalg_ijkl_ipl_kpj_x_test>
		m_utf_linalg_ijkl_ipl_kpj_x;

public:
	//!	Creates the suite
	libtensor_linalg_suite();
};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_LINALG_SUITE_H

