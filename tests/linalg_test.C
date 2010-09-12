#include <cmath>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_test.h"

namespace libtensor {


void linalg_test::perform() throw(libtest::test_exception) {

	test_x_p_p(1, 1, 1);
	test_x_p_p(2, 1, 1);
	test_x_p_p(16, 1, 1);
	test_x_p_p(17, 1, 1);
	test_x_p_p(2, 2, 3);
	test_x_p_p(2, 3, 2);

	test_i_i_x(1, 1, 1);
	test_i_i_x(2, 1, 1);
	test_i_i_x(16, 1, 1);
	test_i_i_x(17, 1, 1);
	test_i_i_x(2, 2, 3);
	test_i_i_x(2, 3, 2);

	test_i_ip_p(1, 1, 1, 1, 1);
	test_i_ip_p(1, 2, 2, 1, 1);
	test_i_ip_p(2, 1, 1, 1, 1);
	test_i_ip_p(16, 16, 16, 1, 1);
	test_i_ip_p(17, 3, 3, 1, 1);
	test_i_ip_p(2, 2, 2, 3, 4);
	test_i_ip_p(2, 2, 4, 3, 2);

	test_i_pi_p(1, 1, 1, 1, 1);
	test_i_pi_p(1, 2, 1, 1, 1);
	test_i_pi_p(2, 1, 1, 2, 1);
	test_i_pi_p(16, 16, 1, 16, 1);
	test_i_pi_p(17, 3, 1, 17, 1);
	test_i_pi_p(2, 2, 2, 3, 4);
	test_i_pi_p(2, 2, 4, 3, 2);

	test_ij_i_j(1, 1, 1, 1, 1);
	test_ij_i_j(1, 2, 1, 2, 1);
	test_ij_i_j(2, 1, 1, 1, 1);
	test_ij_i_j(16, 16, 1, 16, 1);
	test_ij_i_j(3, 17, 1, 17, 1);
	test_ij_i_j(2, 2, 2, 3, 4);
	test_ij_i_j(2, 2, 4, 3, 2);

	test_ij_ip_jp(1, 1, 1, 1, 1, 1);
	test_ij_ip_jp(1, 2, 3, 3, 2, 3);
	test_ij_ip_jp(2, 1, 3, 3, 1, 3);
	test_ij_ip_jp(16, 16, 1, 1, 16, 1);
	test_ij_ip_jp(3, 17, 2, 2, 17, 2);
	test_ij_ip_jp(2, 2, 2, 2, 3, 4);
	test_ij_ip_jp(2, 2, 2, 4, 3, 2);

	test_ij_ip_pj(1, 1, 1, 1, 1, 1);
	test_ij_ip_pj(1, 2, 3, 3, 2, 2);
	test_ij_ip_pj(2, 1, 3, 3, 1, 1);
	test_ij_ip_pj(16, 16, 1, 1, 16, 16);
	test_ij_ip_pj(3, 17, 2, 2, 17, 17);
	test_ij_ip_pj(2, 2, 2, 2, 3, 4);
	test_ij_ip_pj(2, 2, 2, 4, 3, 2);

	test_ij_pi_jp(1, 1, 1, 1, 1, 1);
	test_ij_pi_jp(1, 2, 3, 2, 3, 1);
	test_ij_pi_jp(2, 1, 3, 1, 3, 2);
	test_ij_pi_jp(16, 16, 1, 16, 1, 16);
	test_ij_pi_jp(3, 17, 2, 17, 2, 3);
	test_ij_pi_jp(2, 2, 2, 2, 3, 4);
	test_ij_pi_jp(2, 2, 2, 4, 3, 2);

	test_ij_pi_pj(1, 1, 1, 1, 1, 1);
	test_ij_pi_pj(1, 2, 3, 2, 1, 2);
	test_ij_pi_pj(2, 1, 3, 1, 2, 1);
	test_ij_pi_pj(16, 16, 1, 16, 16, 16);
	test_ij_pi_pj(3, 17, 2, 17, 3, 17);
	test_ij_pi_pj(2, 2, 2, 2, 3, 4);
	test_ij_pi_pj(2, 2, 2, 4, 3, 2);

	test_x_pq_qp(1, 1, 1, 1);
	test_x_pq_qp(1, 2, 2, 1);
	test_x_pq_qp(2, 1, 1, 2);
	test_x_pq_qp(2, 2, 2, 2);
	test_x_pq_qp(16, 2, 2, 16);
	test_x_pq_qp(17, 3, 3, 17);
	test_x_pq_qp(2, 2, 3, 4);
	test_x_pq_qp(2, 2, 4, 3);

	//            ni  np  nq  sia sic spa sqb
	test_i_ipq_qp(1,  1,  1,  1,  1,  1,  1);
	test_i_ipq_qp(1,  1,  2,  2,  1,  2,  1);
	test_i_ipq_qp(1,  2,  1,  2,  1,  1,  2);
	test_i_ipq_qp(2,  1,  1,  1,  2,  1,  1);
	test_i_ipq_qp(2,  2,  2,  4,  2,  2,  2);
	test_i_ipq_qp(5,  3,  7,  21, 5,  7,  3);
	test_i_ipq_qp(16, 16, 16, 256,16, 16, 16);
	test_i_ipq_qp(17, 9,  5,  50, 20, 5,  10);

	//              ni  nj  np  nq  sia sic sjb spa sqb
	test_ij_ipq_jqp(1,  1,  1,  1,  1,  1,  1,  1,  1);
	test_ij_ipq_jqp(1,  1,  2,  1,  2,  1,  2,  1,  2);
	test_ij_ipq_jqp(1,  2,  1,  2,  2,  2,  2,  2,  1);
	test_ij_ipq_jqp(2,  1,  1,  1,  1,  1,  1,  1,  1);
	test_ij_ipq_jqp(2,  2,  2,  2,  4,  2,  4,  2,  2);
	test_ij_ipq_jqp(5,  3,  7,  4,  28, 3,  28, 4,  7);
	test_ij_ipq_jqp(16, 16, 16, 16, 256,16, 256,16, 16);
	test_ij_ipq_jqp(17, 9,  5,  2,  60, 10, 30, 10, 10);

	test_ijkl_ipl_kpj(1, 1, 1, 1, 1);
	test_ijkl_ipl_kpj(2, 1, 1, 1, 1);
	test_ijkl_ipl_kpj(1, 2, 1, 1, 1);
	test_ijkl_ipl_kpj(1, 1, 2, 1, 1);
	test_ijkl_ipl_kpj(1, 1, 1, 2, 1);
	test_ijkl_ipl_kpj(1, 1, 1, 1, 2);
	test_ijkl_ipl_kpj(2, 3, 2, 3, 2);
	test_ijkl_ipl_kpj(3, 5, 1, 7, 13);
	test_ijkl_ipl_kpj(16, 16, 16, 16, 16);
	test_ijkl_ipl_kpj(17, 16, 17, 16, 17);

	test_ijkl_iplq_kpjq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_kpjq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_kpjq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_kpjq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_kpjq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_kpjq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_kpjq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_kpjq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_iplq_pkjq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkjq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkjq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkjq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkjq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkjq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkjq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_pkjq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_iplq_pkqj_x(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj_x(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj_x(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj_x(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkqj_x(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkqj_x(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkqj_x(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkqj_x(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkqj_x(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkqj_x(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_pkqj_x(17, 16, 17, 16, 17, 16);

	test_ijkl_ipql_pkqj_x(1, 1, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj_x(2, 1, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj_x(1, 2, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj_x(1, 1, 2, 1, 1, 1);
	test_ijkl_ipql_pkqj_x(1, 1, 1, 2, 1, 1);
	test_ijkl_ipql_pkqj_x(1, 1, 1, 1, 2, 1);
	test_ijkl_ipql_pkqj_x(1, 1, 1, 1, 1, 2);
	test_ijkl_ipql_pkqj_x(2, 3, 2, 3, 2, 3);
	test_ijkl_ipql_pkqj_x(3, 5, 1, 7, 13, 11);
	test_ijkl_ipql_pkqj_x(16, 16, 16, 16, 16, 16);
	test_ijkl_ipql_pkqj_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pilq_kpjq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_kpjq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_kpjq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_kpjq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_kpjq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_kpjq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_kpjq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pilq_kpjq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pilq_pkjq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_pkjq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_pkjq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_pkjq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_pkjq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_pkjq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_pkjq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pilq_pkjq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_piql_kpqj_x(1, 1, 1, 1, 1, 1);
	test_ijkl_piql_kpqj_x(2, 1, 1, 1, 1, 1);
	test_ijkl_piql_kpqj_x(1, 2, 1, 1, 1, 1);
	test_ijkl_piql_kpqj_x(1, 1, 2, 1, 1, 1);
	test_ijkl_piql_kpqj_x(1, 1, 1, 2, 1, 1);
	test_ijkl_piql_kpqj_x(1, 1, 1, 1, 2, 1);
	test_ijkl_piql_kpqj_x(1, 1, 1, 1, 1, 2);
	test_ijkl_piql_kpqj_x(2, 3, 2, 3, 2, 3);
	test_ijkl_piql_kpqj_x(3, 5, 1, 7, 13, 11);
	test_ijkl_piql_kpqj_x(16, 16, 16, 16, 16, 16);
	test_ijkl_piql_kpqj_x(17, 16, 17, 16, 17, 16);

	test_ijkl_piql_pkqj_x(1, 1, 1, 1, 1, 1);
	test_ijkl_piql_pkqj_x(2, 1, 1, 1, 1, 1);
	test_ijkl_piql_pkqj_x(1, 2, 1, 1, 1, 1);
	test_ijkl_piql_pkqj_x(1, 1, 2, 1, 1, 1);
	test_ijkl_piql_pkqj_x(1, 1, 1, 2, 1, 1);
	test_ijkl_piql_pkqj_x(1, 1, 1, 1, 2, 1);
	test_ijkl_piql_pkqj_x(1, 1, 1, 1, 1, 2);
	test_ijkl_piql_pkqj_x(2, 3, 2, 3, 2, 3);
	test_ijkl_piql_pkqj_x(3, 5, 1, 7, 13, 11);
	test_ijkl_piql_pkqj_x(16, 16, 16, 16, 16, 16);
	test_ijkl_piql_pkqj_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pkiq_jplq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_jplq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_jplq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pkiq_jplq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pkiq_jplq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pkiq_jplq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pkiq_jplq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pkiq_jplq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pkiq_jplq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pkiq_jplq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pkiq_jplq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_jpql_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_jpql_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pkiq_jpql_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pkiq_jpql_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pkiq_jpql_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pkiq_jpql_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pkiq_jpql_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pkiq_jpql_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pkiq_jpql_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pkiq_pjlq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_pjlq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_pjlq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pkiq_pjlq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pkiq_pjlq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pkiq_pjlq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pkiq_pjlq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pkiq_pjlq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pkiq_pjlq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pkiq_pjlq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pkiq_pjlq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pkiq_pjql_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_pjql_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pkiq_pjql_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pkiq_pjql_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pkiq_pjql_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pkiq_pjql_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pkiq_pjql_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pkiq_pjql_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pkiq_pjql_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pkiq_pjql_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pkiq_pjql_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pliq_jpkq_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pliq_jpkq_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pliq_jpkq_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pliq_jpkq_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pliq_jpkq_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pliq_jpkq_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pliq_jpkq_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pliq_jpkq_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pliq_jpkq_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pliq_jpkq_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pliq_jpkq_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pliq_jpqk_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pliq_jpqk_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pliq_jpqk_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pliq_jpqk_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pliq_jpqk_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pliq_jpqk_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pliq_jpqk_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pliq_jpqk_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pliq_jpqk_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pliq_jpqk_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pliq_jpqk_x(17, 16, 17, 16, 17, 16);

	test_ijkl_pliq_pjqk_x(1, 1, 1, 1, 1, 1);
	test_ijkl_pliq_pjqk_x(2, 1, 1, 1, 1, 1);
	test_ijkl_pliq_pjqk_x(1, 2, 1, 1, 1, 1);
	test_ijkl_pliq_pjqk_x(1, 1, 2, 1, 1, 1);
	test_ijkl_pliq_pjqk_x(1, 1, 1, 2, 1, 1);
	test_ijkl_pliq_pjqk_x(1, 1, 1, 1, 2, 1);
	test_ijkl_pliq_pjqk_x(1, 1, 1, 1, 1, 2);
	test_ijkl_pliq_pjqk_x(2, 3, 2, 3, 2, 3);
	test_ijkl_pliq_pjqk_x(3, 5, 1, 7, 13, 11);
	test_ijkl_pliq_pjqk_x(16, 16, 16, 16, 16, 16);
	test_ijkl_pliq_pjqk_x(17, 16, 17, 16, 17, 16);

}


void linalg_test::test_x_p_p(size_t np, size_t spa, size_t spb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_x_p_p(" << np << ", " << spa << ", " << spb
		<< ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0;

	try {

	size_t sza = np * spa, szb = np * spb;

	a = new double[sza];
	b = new double[szb];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();

	double c = linalg::x_p_p(np, a, spa, b, spb);
	double c_ref = linalg_base_generic::x_p_p(np, a, spa, b, spb);

	if(!cmp(c - c_ref, c_ref)) {
		fail_test(tnss.c_str(), __FILE__, __LINE__,
			"Incorrect result.");
	}

	delete [] a; a = 0;
	delete [] b; b = 0;

	} catch(exception &e) {
		delete [] a;
		delete [] b;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a;
		delete [] b;
		throw;
	}
}


void linalg_test::test_i_i_x(size_t ni, size_t sia, size_t sic)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_i_x(" << ni << ", " << sia << ", " << sic
		<< ")";
	std::string tnss = ss.str();

	double *a = 0, *c = 0, *c_ref = 0;
	double b = 0.0;

	try {

	size_t sza = ni * sia, szc = ni * sic;

	a = new double[sza];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();
	b = drand48();

	linalg::i_i_x(ni, a, sia, b, c, sic);
	linalg_base_generic::i_i_x(ni, a, sia, b, c_ref, sic);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result.");
		}
	}

	delete [] a; a = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_i_ip_p(size_t ni, size_t np, size_t sia, size_t sic,
	size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_ip_p(" << ni << ", " << np << ", " << sia
		<< ", " << sic << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::i_ip_p_x(ni, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::i_ip_p_x(ni, np, a, sia, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::i_ip_p_x(ni, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::i_ip_p_x(ni, np, a, sia, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::i_ip_p_x(ni, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::i_ip_p_x(ni, np, a, sia, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::i_ip_p_x(ni, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::i_ip_p_x(ni, np, a, sia, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::i_ip_p_x(ni, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::i_ip_p_x(ni, np, a, sia, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_i_pi_p(size_t ni, size_t np, size_t sic, size_t spa,
	size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_pi_p(" << ni << ", " << np << ", " << sic
		<< ", " << spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::i_pi_p_x(ni, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::i_pi_p_x(ni, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::i_pi_p_x(ni, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::i_pi_p_x(ni, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::i_pi_p_x(ni, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_i_j(size_t ni, size_t nj, size_t sia, size_t sic,
	size_t sjb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_i_j(" << ni << ", " << nj << ", " << sia
		<< ", " << sic << ", " << sjb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_i_j_x(ni, nj, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_i_j_x(ni, nj, a, sia, b, sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_i_j_x(ni, nj, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_i_j_x(ni, nj, a, sia, b, sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_i_j_x(ni, nj, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_i_j_x(ni, nj, a, sia, b, sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_i_j_x(ni, nj, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_i_j_x(ni, nj, a, sia, b, sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_i_j_x(ni, nj, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_i_j_x(ni, nj, a, sia, b, sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_ip_jp(size_t ni, size_t nj, size_t np, size_t sia,
	size_t sic, size_t sjb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ip_jp(" << ni << ", " << nj << ", " << np
		<< ", " << sia << ", " << sic << ", " << sjb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c, sic, d);
	linalg_base_generic::ij_ip_jp_x(ni, nj, np, a, sia, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_ip_pj(size_t ni, size_t nj, size_t np, size_t sia,
	size_t sic, size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ip_pj(" << ni << ", " << nj << ", " << np
		<< ", " << sia << ", " << sic << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c, sic, d);
	linalg_base_generic::ij_ip_pj_x(ni, nj, np, a, sia, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_pi_jp(size_t ni, size_t nj, size_t np, size_t sic,
	size_t sjb, size_t spa) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_pi_jp(" << ni << ", " << nj << ", " << np
		<< ", " << sic << ", " << sjb << ", " << spa << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * spa, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c, sic, d);
	linalg_base_generic::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c, sic, d);
	linalg_base_generic::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c, sic, d);
	linalg_base_generic::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c, sic, d);
	linalg_base_generic::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c, sic, d);
	linalg_base_generic::ij_pi_jp_x(ni, nj, np, a, spa, b, sjb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_pi_pj(size_t ni, size_t nj, size_t np, size_t sic,
	size_t spa, size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_pi_pj(" << ni << ", " << nj << ", " << np
		<< ", " << sic << ", " << spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}

void linalg_test::test_x_pq_qp(size_t np, size_t nq, size_t spa, size_t sqb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_x_pq_qp(" << np << ", " << nq << ", " << spa
		<< ", " << sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0;

	try {

	size_t sza = np * spa, szb = nq * sqb;

	a = new double[sza];
	b = new double[szb];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();

	double c = linalg::x_pq_qp(np, nq, a, spa, b, sqb);
	double c_ref = linalg_base_generic::x_pq_qp(np, nq, a, spa, b, sqb);

	if(!cmp(c - c_ref, c_ref)) {
		fail_test(tnss.c_str(), __FILE__, __LINE__,
			"Incorrect result.");
	}

	delete [] a; a = 0;
	delete [] b; b = 0;

	} catch(exception &e) {
		delete [] a;
		delete [] b;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a;
		delete [] b;
		throw;
	}
}


void linalg_test::test_i_ipq_qp(size_t ni, size_t np, size_t nq, size_t sia,
	size_t sic, size_t spa, size_t sqb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_ipq_qp(" << ni << ", " << np << ", " << nq
		<< ", " << sia << ", " << sic << ", " << spa << ", "
		<< sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = nq * sqb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c, sic, d);
	linalg_base_generic::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c, sic, d);
	linalg_base_generic::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c, sic, d);
	linalg_base_generic::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c, sic, d);
	linalg_base_generic::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c, sic, d);
	linalg_base_generic::i_ipq_qp_x(ni, np, nq, a, spa, sia, b, sqb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ipq_jqp(" << ni << ", " << nj << ", "
		<< np << ", " << nq << ", " << sia << ", " << sic << ", "
		<< sjb << ", " << spa << ", " << sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
		d);
	linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
		sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
		d);
	linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
		sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
		d);
	linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
		sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
		d);
	linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
		sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb, sjb, c, sic,
		d);
	linalg_base_generic::ij_ipq_jqp_x(ni, nj, np, nq, a, spa, sia, b, sqb,
		sjb, c_ref, sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_ipl_kpj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_ipl_kpj(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nl, szb = nk * np * nj, szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c, d);
	linalg_base_generic::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c, d);
	linalg_base_generic::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c, d);
	linalg_base_generic::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c, d);
	linalg_base_generic::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c, d);
	linalg_base_generic::ijkl_ipl_kpj_x(ni, nj, nk, nl, np, a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_iplq_kpjq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_kpjq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_iplq_pkjq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_pkjq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_iplq_pkqj_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_pkqj_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nq * nj,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_iplq_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_ipql_pkqj_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_ipql_pkqj_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nq * nl, szb = np * nk * nq * nj,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -0.0;
	linalg::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_ipql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pilq_kpjq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pilq_kpjq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * ni * nl * nq, szb = nk * np * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_kpjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pilq_pkjq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pilq_pkjq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * ni * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pilq_pkjq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_piql_kpqj_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_piql_kpqj_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * ni * nq * nl, szb = nk * np * nq * nj,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_kpqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_piql_pkqj_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_piql_pkqj_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * ni * nq * nl, szb = np * nk * nq * nj,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_piql_pkqj_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pkiq_jplq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pkiq_jplq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nk * ni * nq, szb = nj * np * nl * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jplq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pkiq_jpql_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pkiq_jpql_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nk * ni * nq, szb = nj * np * nq * nl,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_jpql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pkiq_pjlq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pkiq_pjlq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nk * ni * nq, szb = np * nj * nl * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjlq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pkiq_pjql_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pkiq_pjql_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nk * ni * nq, szb = np * nj * nq * nl,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pkiq_pjql_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pliq_jpkq_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pliq_jpkq_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nl * ni * nq, szb = nj * np * nk * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpkq_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pliq_jpqk_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pliq_jpqk_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nl * ni * nq, szb = nj * np * nq * nk,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_jpqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_ijkl_pliq_pjqk_x(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pliq_pjqk_x(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * nl * ni * nq, szb = np * nj * nq * nk,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq, a, b, c, d);
	linalg_base_generic::ijkl_pliq_pjqk_x(ni, nj, nk, nl, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


bool linalg_test::cmp(double diff, double ref) {

	static const double k_thresh = 1e-12;

	if(fabs(ref) > 1.0) return fabs(diff) < fabs(ref) * k_thresh;
	else return fabs(diff) < k_thresh;
}


} // namespace libtensor
