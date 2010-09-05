#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libvmm/std_allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_contract2.h>
#include "compare_ref.h"
#include "tod_contract2_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

const double tod_contract2_test::k_thresh = 1e-14;

void tod_contract2_test::perform() throw(libtest::test_exception) {

	//
	//	Test one-index contractions
	//

	test_0_p_p(1);
	test_0_p_p(2);
	test_0_p_p(5);
	test_0_p_p(16);
	test_0_p_p(1, -0.5);
	test_0_p_p(2, -2.0);
	test_0_p_p(5, 1.2);
	test_0_p_p(16, 0.7);

	test_i_p_pi(1, 1);
	test_i_p_pi(1, 2);
	test_i_p_pi(2, 1);
	test_i_p_pi(3, 3);
	test_i_p_pi(3, 5);
	test_i_p_pi(16, 16);
	test_i_p_pi(1, 1, -0.5);
	test_i_p_pi(1, 2, 2.0);
	test_i_p_pi(2, 1, -1.0);
	test_i_p_pi(3, 3, 3.7);
	test_i_p_pi(3, 5, 1.0);
	test_i_p_pi(16, 16, 0.7);

	test_i_p_ip(1, 1);
	test_i_p_ip(1, 2);
	test_i_p_ip(2, 1);
	test_i_p_ip(3, 3);
	test_i_p_ip(3, 5);
	test_i_p_ip(16, 16);
	test_i_p_ip(1, 1, -0.5);
	test_i_p_ip(1, 2, 2.0);
	test_i_p_ip(2, 1, -1.0);
	test_i_p_ip(3, 3, 3.7);
	test_i_p_ip(3, 5, 1.0);
	test_i_p_ip(16, 16, 0.7);

	test_i_pi_p(1, 1);
	test_i_pi_p(1, 2);
	test_i_pi_p(2, 1);
	test_i_pi_p(3, 3);
	test_i_pi_p(3, 5);
	test_i_pi_p(16, 16);
	test_i_pi_p(1, 1, -0.5);
	test_i_pi_p(1, 2, 2.0);
	test_i_pi_p(2, 1, -1.0);
	test_i_pi_p(3, 3, 3.7);
	test_i_pi_p(3, 5, 1.0);
	test_i_pi_p(16, 16, 0.7);

	test_i_ip_p(1, 1);
	test_i_ip_p(1, 2);
	test_i_ip_p(2, 1);
	test_i_ip_p(3, 3);
	test_i_ip_p(3, 5);
	test_i_ip_p(16, 16);
	test_i_ip_p(1, 1, -0.5);
	test_i_ip_p(1, 2, 2.0);
	test_i_ip_p(2, 1, -1.0);
	test_i_ip_p(3, 3, 3.7);
	test_i_ip_p(3, 5, 1.0);
	test_i_ip_p(16, 16, 0.7);

	test_ij_pi_pj(1, 1, 1);
	test_ij_pi_pj(1, 1, 2);
	test_ij_pi_pj(1, 2, 1);
	test_ij_pi_pj(2, 1, 1);
	test_ij_pi_pj(3, 3, 3);
	test_ij_pi_pj(3, 5, 7);
	test_ij_pi_pj(16, 16, 16);
	test_ij_pi_pj(1, 1, 1, -0.5);
	test_ij_pi_pj(1, 1, 2, 2.0);
	test_ij_pi_pj(1, 2, 1, -1.0);
	test_ij_pi_pj(2, 1, 1, 3.7);
	test_ij_pi_pj(3, 3, 3, 1.0);
	test_ij_pi_pj(3, 5, 7, -1.2);
	test_ij_pi_pj(16, 16, 16, 0.7);

	test_ij_pi_jp(1, 1, 1);
	test_ij_pi_jp(1, 1, 2);
	test_ij_pi_jp(1, 2, 1);
	test_ij_pi_jp(2, 1, 1);
	test_ij_pi_jp(3, 3, 3);
	test_ij_pi_jp(3, 5, 7);
	test_ij_pi_jp(16, 16, 16);
	test_ij_pi_jp(1, 1, 1, -0.5);
	test_ij_pi_jp(1, 1, 2, 2.0);
	test_ij_pi_jp(1, 2, 1, -1.0);
	test_ij_pi_jp(2, 1, 1, 3.7);
	test_ij_pi_jp(3, 3, 3, 1.0);
	test_ij_pi_jp(3, 5, 7, -1.2);
	test_ij_pi_jp(16, 16, 16, 0.7);

	test_ij_ip_pj(1, 1, 1);
	test_ij_ip_pj(1, 1, 2);
	test_ij_ip_pj(1, 2, 1);
	test_ij_ip_pj(2, 1, 1);
	test_ij_ip_pj(3, 3, 3);
	test_ij_ip_pj(3, 5, 7);
	test_ij_ip_pj(16, 16, 16);
	test_ij_ip_pj(1, 1, 1, -0.5);
	test_ij_ip_pj(1, 1, 2, 2.0);
	test_ij_ip_pj(1, 2, 1, -1.0);
	test_ij_ip_pj(2, 1, 1, 3.7);
	test_ij_ip_pj(3, 3, 3, 1.0);
	test_ij_ip_pj(3, 5, 7, -1.2);
	test_ij_ip_pj(16, 16, 16, 0.7);

	test_ij_ip_jp(1, 1, 1);
	test_ij_ip_jp(1, 1, 2);
	test_ij_ip_jp(1, 2, 1);
	test_ij_ip_jp(2, 1, 1);
	test_ij_ip_jp(3, 3, 3);
	test_ij_ip_jp(3, 5, 7);
	test_ij_ip_jp(16, 16, 16);
	test_ij_ip_jp(1, 1, 1, -0.5);
	test_ij_ip_jp(1, 1, 2, 2.0);
	test_ij_ip_jp(1, 2, 1, -1.0);
	test_ij_ip_jp(2, 1, 1, 3.7);
	test_ij_ip_jp(3, 3, 3, 1.0);
	test_ij_ip_jp(3, 5, 7, -1.2);
	test_ij_ip_jp(16, 16, 16, 0.7);

	test_ij_pj_pi(1, 1, 1);
	test_ij_pj_pi(1, 1, 2);
	test_ij_pj_pi(1, 2, 1);
	test_ij_pj_pi(2, 1, 1);
	test_ij_pj_pi(3, 3, 3);
	test_ij_pj_pi(3, 5, 7);
	test_ij_pj_pi(16, 16, 16);
	test_ij_pj_pi(1, 1, 1, -0.5);
	test_ij_pj_pi(1, 1, 2, 2.0);
	test_ij_pj_pi(1, 2, 1, -1.0);
	test_ij_pj_pi(2, 1, 1, 3.7);
	test_ij_pj_pi(3, 3, 3, 1.0);
	test_ij_pj_pi(3, 5, 7, -1.2);
	test_ij_pj_pi(16, 16, 16, 0.7);

	test_ij_pj_ip(1, 1, 1);
	test_ij_pj_ip(1, 1, 2);
	test_ij_pj_ip(1, 2, 1);
	test_ij_pj_ip(2, 1, 1);
	test_ij_pj_ip(3, 3, 3);
	test_ij_pj_ip(3, 5, 7);
	test_ij_pj_ip(16, 16, 16);
	test_ij_pj_ip(1, 1, 1, -0.5);
	test_ij_pj_ip(1, 1, 2, 2.0);
	test_ij_pj_ip(1, 2, 1, -1.0);
	test_ij_pj_ip(2, 1, 1, 3.7);
	test_ij_pj_ip(3, 3, 3, 1.0);
	test_ij_pj_ip(3, 5, 7, -1.2);
	test_ij_pj_ip(16, 16, 16, 0.7);

	test_ij_jp_ip(1, 1, 1);
	test_ij_jp_ip(1, 1, 2);
	test_ij_jp_ip(1, 2, 1);
	test_ij_jp_ip(2, 1, 1);
	test_ij_jp_ip(3, 3, 3);
	test_ij_jp_ip(3, 5, 7);
	test_ij_jp_ip(16, 16, 16);
	test_ij_jp_ip(1, 1, 1, -0.5);
	test_ij_jp_ip(1, 1, 2, 2.0);
	test_ij_jp_ip(1, 2, 1, -1.0);
	test_ij_jp_ip(2, 1, 1, 3.7);
	test_ij_jp_ip(3, 3, 3, 1.0);
	test_ij_jp_ip(3, 5, 7, -1.2);
	test_ij_jp_ip(16, 16, 16, 0.7);

	test_ij_jp_pi(1, 1, 1);
	test_ij_jp_pi(1, 1, 2);
	test_ij_jp_pi(1, 2, 1);
	test_ij_jp_pi(2, 1, 1);
	test_ij_jp_pi(3, 3, 3);
	test_ij_jp_pi(3, 5, 7);
	test_ij_jp_pi(16, 16, 16);
	test_ij_jp_pi(1, 1, 1, -0.5);
	test_ij_jp_pi(1, 1, 2, 2.0);
	test_ij_jp_pi(1, 2, 1, -1.0);
	test_ij_jp_pi(2, 1, 1, 3.7);
	test_ij_jp_pi(3, 3, 3, 1.0);
	test_ij_jp_pi(3, 5, 7, -1.2);
	test_ij_jp_pi(16, 16, 16, 0.7);

	//
	//	Test two-index contractions
	//

	test_ij_pqi_pjq(1, 1, 1, 1);
	test_ij_pqi_pjq(1, 1, 1, 2);
	test_ij_pqi_pjq(1, 1, 2, 1);
	test_ij_pqi_pjq(1, 2, 1, 1);
	test_ij_pqi_pjq(2, 1, 1, 1);
	test_ij_pqi_pjq(3, 3, 3, 3);
	test_ij_pqi_pjq(11, 5, 7, 3);
	test_ij_pqi_pjq(16, 16, 16, 16);
	test_ij_pqi_pjq(1, 1, 1, 1, -0.5);
	test_ij_pqi_pjq(1, 1, 1, 2, 2.0);
	test_ij_pqi_pjq(1, 1, 2, 1, -1.0);
	test_ij_pqi_pjq(1, 2, 1, 1, 3.7);
	test_ij_pqi_pjq(2, 1, 1, 1, 1.0);
	test_ij_pqi_pjq(3, 3, 3, 3, 1.0);
	test_ij_pqi_pjq(11, 5, 7, 3, -1.2);
	test_ij_pqi_pjq(16, 16, 16, 16, 0.7);

	test_ij_ipq_jqp(1, 1, 1, 1);
	test_ij_ipq_jqp(1, 1, 1, 2);
	test_ij_ipq_jqp(1, 1, 2, 1);
	test_ij_ipq_jqp(1, 2, 1, 1);
	test_ij_ipq_jqp(2, 1, 1, 1);
	test_ij_ipq_jqp(3, 3, 3, 3);
	test_ij_ipq_jqp(11, 5, 7, 3);
	test_ij_ipq_jqp(16, 16, 16, 16);
	test_ij_ipq_jqp(1, 1, 1, 1, -0.5);
	test_ij_ipq_jqp(1, 1, 1, 2, 2.0);
	test_ij_ipq_jqp(1, 1, 2, 1, -1.0);
	test_ij_ipq_jqp(1, 2, 1, 1, 3.7);
	test_ij_ipq_jqp(2, 1, 1, 1, 1.0);
	test_ij_ipq_jqp(3, 3, 3, 3, 1.0);
	test_ij_ipq_jqp(11, 5, 7, 3, -1.2);
	test_ij_ipq_jqp(16, 16, 16, 16, 0.7);

	test_ij_jpq_iqp(1, 1, 1, 1);
	test_ij_jpq_iqp(1, 1, 1, 2);
	test_ij_jpq_iqp(1, 1, 2, 1);
	test_ij_jpq_iqp(1, 2, 1, 1);
	test_ij_jpq_iqp(2, 1, 1, 1);
	test_ij_jpq_iqp(3, 3, 3, 3);
	test_ij_jpq_iqp(11, 5, 7, 3);
	test_ij_jpq_iqp(16, 16, 16, 16);
	test_ij_jpq_iqp(1, 1, 1, 1, -0.5);
	test_ij_jpq_iqp(1, 1, 1, 2, 2.0);
	test_ij_jpq_iqp(1, 1, 2, 1, -1.0);
	test_ij_jpq_iqp(1, 2, 1, 1, 3.7);
	test_ij_jpq_iqp(2, 1, 1, 1, 1.0);
	test_ij_jpq_iqp(3, 3, 3, 3, 1.0);
	test_ij_jpq_iqp(11, 5, 7, 3, -1.2);
	test_ij_jpq_iqp(16, 16, 16, 16, 0.7);

	test_ij_pq_ijpq(1, 1, 1, 1);
	test_ij_pq_ijpq(2, 2, 2, 2);
	test_ij_pq_ijpq_a(1, 1, 1, 1, 0.25);
	test_ij_pq_ijpq_a(2, 2, 2, 2, 0.25);

	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_kpjq(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_kpjq(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_kpjq(3, 5, 2, 7, 13, 11);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_iplq_kpjq(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_iplq_kpjq(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_iplq_kpjq(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_iplq_kpjq(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_iplq_kpjq(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_iplq_kpjq(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_iplq_kpjq(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkjq(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkjq(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkjq(3, 5, 2, 7, 13, 11);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_iplq_pkjq(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_iplq_pkjq(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_iplq_pkjq(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_iplq_pkjq(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_iplq_pkjq(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_iplq_pkjq(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_iplq_pkjq(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkqj(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkqj(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkqj(3, 5, 2, 7, 13, 11);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_iplq_pkqj(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_iplq_pkqj(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_iplq_pkqj(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_iplq_pkqj(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_iplq_pkqj(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_iplq_pkqj(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_iplq_pkqj(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1);
	test_ijkl_ipql_kpqj(2, 1, 1, 1, 1, 1);
	test_ijkl_ipql_kpqj(1, 2, 1, 1, 1, 1);
	test_ijkl_ipql_kpqj(1, 1, 2, 1, 1, 1);
	test_ijkl_ipql_kpqj(1, 1, 1, 2, 1, 1);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 2, 1);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 2);
	test_ijkl_ipql_kpqj(2, 3, 2, 3, 2, 3);
	test_ijkl_ipql_kpqj(3, 5, 1, 7, 13, 11);
	test_ijkl_ipql_kpqj(3, 5, 2, 7, 13, 11);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_ipql_kpqj(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_ipql_kpqj(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_ipql_kpqj(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_ipql_kpqj(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_ipql_kpqj(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_ipql_kpqj(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_ipql_kpqj(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_ipql_kpqj(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj(2, 1, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj(1, 2, 1, 1, 1, 1);
	test_ijkl_ipql_pkqj(1, 1, 2, 1, 1, 1);
	test_ijkl_ipql_pkqj(1, 1, 1, 2, 1, 1);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 2, 1);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 2);
	test_ijkl_ipql_pkqj(2, 3, 2, 3, 2, 3);
	test_ijkl_ipql_pkqj(3, 5, 1, 7, 13, 11);
	test_ijkl_ipql_pkqj(3, 5, 2, 7, 13, 11);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_ipql_pkqj(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_ipql_pkqj(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_ipql_pkqj(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_ipql_pkqj(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_ipql_pkqj(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_ipql_pkqj(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_ipql_pkqj(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_ipql_pkqj(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_kpjq(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_kpjq(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_kpjq(3, 5, 2, 7, 13, 11);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_pilq_kpjq(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_pilq_kpjq(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_pilq_kpjq(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_pilq_kpjq(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_pilq_kpjq(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_pilq_kpjq(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_pilq_kpjq(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_pkjq(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_pkjq(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_pkjq(3, 5, 2, 7, 13, 11);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_pilq_pkjq(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_pilq_pkjq(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_pilq_pkjq(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_pilq_pkjq(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_pilq_pkjq(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_pilq_pkjq(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_pilq_pkjq(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1);
	test_ijkl_piql_kpqj(2, 1, 1, 1, 1, 1);
	test_ijkl_piql_kpqj(1, 2, 1, 1, 1, 1);
	test_ijkl_piql_kpqj(1, 1, 2, 1, 1, 1);
	test_ijkl_piql_kpqj(1, 1, 1, 2, 1, 1);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 2, 1);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 2);
	test_ijkl_piql_kpqj(2, 3, 2, 3, 2, 3);
	test_ijkl_piql_kpqj(3, 5, 1, 7, 13, 11);
	test_ijkl_piql_kpqj(3, 5, 2, 7, 13, 11);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_piql_kpqj(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_piql_kpqj(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_piql_kpqj(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_piql_kpqj(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_piql_kpqj(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_piql_kpqj(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_piql_kpqj(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_piql_kpqj(3, 5, 2, 7, 13, 11, -1.25);

	test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1);
	test_ijkl_piql_pkqj(2, 1, 1, 1, 1, 1);
	test_ijkl_piql_pkqj(1, 2, 1, 1, 1, 1);
	test_ijkl_piql_pkqj(1, 1, 2, 1, 1, 1);
	test_ijkl_piql_pkqj(1, 1, 1, 2, 1, 1);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 2, 1);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 2);
	test_ijkl_piql_pkqj(2, 3, 2, 3, 2, 3);
	test_ijkl_piql_pkqj(3, 5, 1, 7, 13, 11);
	test_ijkl_piql_pkqj(3, 5, 2, 7, 13, 11);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1, 0.0);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 1, -0.5);
	test_ijkl_piql_pkqj(2, 1, 1, 1, 1, 1, 2.0);
	test_ijkl_piql_pkqj(1, 2, 1, 1, 1, 1, -1.0);
	test_ijkl_piql_pkqj(1, 1, 2, 1, 1, 1, 3.7);
	test_ijkl_piql_pkqj(1, 1, 1, 2, 1, 1, 1.0);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 2, 1, -1.2);
	test_ijkl_piql_pkqj(1, 1, 1, 1, 1, 2, 0.7);
	test_ijkl_piql_pkqj(2, 3, 2, 3, 2, 3, 12.3);
	test_ijkl_piql_pkqj(3, 5, 1, 7, 13, 11, -1.25);
	test_ijkl_piql_pkqj(3, 5, 2, 7, 13, 11, -1.25);

	test_ij_ipqr_jpqr(3, 4, 5, 6, 7);
	test_ij_ipqr_jpqr_a(3, 4, 5, 6, 7, -2.0);

	test_ij_jpqr_iprq(3, 3, 3, 2, 4, 1.0);

	test_ij_pqir_pqjr(3, 4, 5, 6, 7);
	test_ij_pqir_pqjr_a(3, 4, 5, 6, 7, 2.0);
	test_ij_pqir_pqjr(3, 3, 3, 3, 3);
	test_ij_pqir_pqjr(3, 1, 3, 1, 2);
	test_ij_pqir_pqjr(3, 3, 1, 1, 2);

	test_ijkl_pi_jklp(1, 4, 5, 6, 2);
	test_ijkl_pi_jklp(3, 4, 5, 6, 7);
	test_ijkl_pi_jklp(10, 10, 10, 10, 6);
	test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, 1.0);
	test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, 0.0);
	test_ijkl_pi_jklp_a(3, 4, 5, 6, 7, -2.0);

	test_jikl_pi_jpkl(1, 4, 5, 6, 2);
	test_jikl_pi_jpkl(3, 4, 5, 6, 7);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, 0.0);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, -2.0);

	test_ijkl_ijp_klp(1, 1, 1, 1, 1);
	test_ijkl_ijp_klp(3, 4, 5, 6, 7);
	test_ijkl_ijp_klp(5, 6, 3, 4, 7);
	test_ijkl_ijp_klp(1, 100, 1, 100, 100);
	test_ijkl_ijp_klp_a(3, 4, 5, 6, 7, -2.0);

	test_ijkl_ij_kl(3, 4, 5, 6);

	test_ijkl_ij_lk(3, 4, 5, 6);

}


void tod_contract2_test::test_0_p_p(size_t np, double d)
	throw(libtest::test_exception) {

	//	c = \sum_p a_p b_p

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_0_p_p(" << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<1> ia1, ia2; ia2[0] = np - 1;
	index<1> ib1, ib2; ib2[0] = np - 1;
	index<0> ic1, ic2;
	dimensions<1> dima(index_range<1>(ia1, ia2));
	dimensions<1> dimb(index_range<1>(ib1, ib2));
	dimensions<0> dimc(index_range<0>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<1, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);
	tensor<0, double, allocator> tc(dimc);
	tensor<0, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<1, double> tca(ta);
	tensor_ctrl<1, double> tcb(tb);
	tensor_ctrl<0, double> tcc(tc);
	tensor_ctrl<0, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<1> ia; index<1> ib; index<0> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t p = 0; p < np; p++) {
		ia[0] = p;
		ib[0] = p;
		abs_index<1> aa(ia, dima), ab(ib, dimb);
		dtc2[0] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	cij_max = fabs(dtc2[0]);

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<0, 0, 1> contr;
	contr.contract(0, 0);
	if(d == 0.0) tod_contract2<0, 0, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<0, 0, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<0>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_i_p_pi(size_t ni, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_i = \sum_p a_p b_{pi}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_i_p_pi(" << ni << ", " << np << ", "
		<< d << ")";
	std::string tns = tnss.str();

	try {

	index<1> ia1, ia2; ia2[0] = np - 1;
	index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
	index<1> ic1, ic2; ic2[0] = ni - 1;
	dimensions<1> dima(index_range<1>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<1> dimc(index_range<1>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<1, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<1, double, allocator> tc(dimc);
	tensor<1, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<1, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<1, double> tcc(tc);
	tensor_ctrl<1, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<1> ia; index<2> ib; index<1> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p;
		ib[0] = p; ib[1] = i;
		ic[0] = i;
		abs_index<1> aa(ia, dima), ac(ic, dimc);
		abs_index<2> ab(ib, dimb);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<0, 1, 1> contr;
	contr.contract(0, 0);
	if(d == 0.0) tod_contract2<0, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<0, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_i_p_ip(size_t ni, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_i = \sum_p a_p b_{ip}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_i_p_ip(" << ni << ", " << np << ", "
		<< d << ")";
	std::string tns = tnss.str();

	try {

	index<1> ia1, ia2; ia2[0] = np - 1;
	index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
	index<1> ic1, ic2; ic2[0] = ni - 1;
	dimensions<1> dima(index_range<1>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<1> dimc(index_range<1>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<1, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<1, double, allocator> tc(dimc);
	tensor<1, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<1, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<1, double> tcc(tc);
	tensor_ctrl<1, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<1> ia; index<2> ib; index<1> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p;
		ib[0] = i; ib[1] = p;
		ic[0] = i;
		abs_index<1> aa(ia, dima), ac(ic, dimc);
		abs_index<2> ab(ib, dimb);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<0, 1, 1> contr;
	contr.contract(0, 1);
	if(d == 0.0) tod_contract2<0, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<0, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_i_pi_p(size_t ni, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_i = \sum_p a_{pi} b_p

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_i_pi_p(" << ni << ", " << np << ", "
		<< d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
	index<1> ib1, ib2; ib2[0] = np - 1;
	index<1> ic1, ic2; ic2[0] = ni - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<1> dimb(index_range<1>(ib1, ib2));
	dimensions<1> dimc(index_range<1>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);
	tensor<1, double, allocator> tc(dimc);
	tensor<1, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<1, double> tcb(tb);
	tensor_ctrl<1, double> tcc(tc);
	tensor_ctrl<1, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<1> ib; index<1> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p; ia[1] = i;
		ib[0] = p;
		ic[0] = i;
		abs_index<2> aa(ia, dima);
		abs_index<1> ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 0, 1> contr;
	contr.contract(0, 0);
	if(d == 0.0) tod_contract2<1, 0, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 0, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_i_ip_p(size_t ni, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_i = \sum_p a_{ip} b_p

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_i_ip_p(" << ni << ", " << np << ", "
		<< d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
	index<1> ib1, ib2; ib2[0] = np - 1;
	index<1> ic1, ic2; ic2[0] = ni - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<1> dimb(index_range<1>(ib1, ib2));
	dimensions<1> dimc(index_range<1>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);
	tensor<1, double, allocator> tc(dimc);
	tensor<1, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<1, double> tcb(tb);
	tensor_ctrl<1, double> tcc(tc);
	tensor_ctrl<1, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<1> ib; index<1> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = i; ia[1] = p;
		ib[0] = p;
		ic[0] = i;
		abs_index<2> aa(ia, dima);
		abs_index<1> ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 0, 1> contr;
	contr.contract(1, 0);
	if(d == 0.0) tod_contract2<1, 0, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 0, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<1>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pi_pj(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{pi} b_{pj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pi_pj(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
	index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p; ia[1] = i;
		ib[0] = p; ib[1] = j;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 1> contr;
	contr.contract(0, 0);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pi_jp(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{pi} b_{jp}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pi_jp(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
	index<2> ib1, ib2; ib2[0] = nj - 1; ib2[1] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p; ia[1] = i;
		ib[0] = j; ib[1] = p;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 1> contr;
	contr.contract(0, 1);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_ip_pj(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{ip} b_{pj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_ip_pj(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
	index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = i; ia[1] = p;
		ib[0] = p; ib[1] = j;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 1> contr;
	contr.contract(1, 0);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_ip_jp(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{ip} b_{jp}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_ip_jp(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1;
	index<2> ib1, ib2; ib2[0] = nj - 1; ib2[1] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = i; ia[1] = p;
		ib[0] = j; ib[1] = p;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 1> contr;
	contr.contract(1, 1);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pj_pi(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{pj} b_{pi}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pj_pi(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
	index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p; ia[1] = j;
		ib[0] = p; ib[1] = i;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	permutation<2> permc; permc.permute(0, 1);
	contraction2<1, 1, 1> contr(permc);
	contr.contract(0, 0);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pj_ip(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{pj} b_{ip}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pj_ip(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = nj - 1;
	index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = p; ia[1] = j;
		ib[0] = i; ib[1] = p;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	permutation<2> permc; permc.permute(0, 1);
	contraction2<1, 1, 1> contr(permc);
	contr.contract(0, 1);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_jp_ip(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{jp} b_{ip}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_jp_ip(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1;
	index<2> ib1, ib2; ib2[0] = ni - 1; ib2[1] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = j; ia[1] = p;
		ib[0] = i; ib[1] = p;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	permutation<2> permc; permc.permute(0, 1);
	contraction2<1, 1, 1> contr(permc);
	contr.contract(1, 1);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_jp_pi(
	size_t ni, size_t nj, size_t np, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_p a_{jp} b_{pi}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_jp_pi(" << ni << ", " << nj
		<< ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<2> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1;
	index<2> ib1, ib2; ib2[0] = np - 1; ib2[1] = ni - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<2> ia; index<2> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
		ia[0] = j; ia[1] = p;
		ib[0] = p; ib[1] = i;
		ic[0] = i; ic[1] = j;
		abs_index<2> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	permutation<2> permc; permc.permute(0, 1);
	contraction2<1, 1, 1> contr(permc);
	contr.contract(1, 0);
	if(d == 0.0) tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 1>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pqi_pjq(
	size_t ni, size_t nj, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_{pq} a_{pqi} b_{pjq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pqi_pjq(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<3> ia1, ia2; ia2[0] = np - 1; ia2[1] = nq - 1; ia2[2] = ni - 1;
	index<3> ib1, ib2; ib2[0] = np - 1; ib2[1] = nj - 1; ib2[2] = nq - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<3> dima(index_range<3>(ia1, ia2));
	dimensions<3> dimb(index_range<3>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima);
	tensor<3, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<3, double> tca(ta);
	tensor_ctrl<3, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<3> ia; index<3> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = p; ia[1] = q; ia[2] = i;
		ib[0] = p; ib[1] = j; ib[2] = q;
		ic[0] = i; ic[1] = j;
		abs_index<3> aa(ia, dima), ab(ib, dimb);
		abs_index<2> ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 2> contr;
	contr.contract(0, 0);
	contr.contract(1, 2);
	if(d == 0.0) tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_ipq_jqp(
	size_t ni, size_t nj, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_{pq} a_{ipq} b_{jqp}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_ipq_jqp(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<3> ia1, ia2; ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1;
	index<3> ib1, ib2; ib2[0] = nj - 1; ib2[1] = nq - 1; ib2[2] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<3> dima(index_range<3>(ia1, ia2));
	dimensions<3> dimb(index_range<3>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima);
	tensor<3, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<3, double> tca(ta);
	tensor_ctrl<3, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<3> ia; index<3> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = q;
		ib[0] = j; ib[1] = q; ib[2] = p;
		ic[0] = i; ic[1] = j;
		abs_index<3> aa(ia, dima), ab(ib, dimb);
		abs_index<2> ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 2> contr;
	contr.contract(1, 2);
	contr.contract(2, 1);
	if(d == 0.0) tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_jpq_iqp(
	size_t ni, size_t nj, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = \sum_{pq} a_{jpq} b_{iqp}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_jpq_iqp(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<3> ia1, ia2; ia2[0] = nj - 1; ia2[1] = np - 1; ia2[2] = nq - 1;
	index<3> ib1, ib2; ib2[0] = ni - 1; ib2[1] = nq - 1; ib2[2] = np - 1;
	index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
	dimensions<3> dima(index_range<3>(ia1, ia2));
	dimensions<3> dimb(index_range<3>(ib1, ib2));
	dimensions<2> dimc(index_range<2>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima);
	tensor<3, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<3, double> tca(ta);
	tensor_ctrl<3, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<3> ia; index<3> ib; index<2> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = j; ia[1] = p; ia[2] = q;
		ib[0] = i; ib[1] = q; ib[2] = p;
		ic[0] = i; ic[1] = j;
		abs_index<3> aa(ia, dima), ab(ib, dimb);
		abs_index<2> ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<1, 1, 2> contr(permutation<2>().permute(0, 1));
	contr.contract(1, 2);
	contr.contract(2, 1);
	if(d == 0.0) tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<1, 1, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_pq_ijpq(size_t ni, size_t nj, size_t np,
	size_t nq) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pq} a_{pq} b_{ijpq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pq_ijpq(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ")";
	std::string tns = tnss.str();

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
	index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
			ia[0]=p; ia[1]=q;
			ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<0, 2, 2> contr(permc);
	contr.contract(0, 2);
	contr.contract(1, 3);

	tod_contract2<0, 2, 2> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np,
	size_t nq, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pq_ijpq_a(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << d << ")";
	std::string tns = tnss.str();

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
	index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
			ia[0]=p; ia[1]=q;
			ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<0, 2, 2> contr(permc);
	contr.contract(0, 2);
	contr.contract(1, 3);

	tod_contract2<0, 2, 2> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ijkl_iplq_kpjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{iplq} b_{kpjq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_iplq_kpjq(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
	index<4> ib1, ib2;
	ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
		ib[0] = k; ib[1] = p; ib[2] = j; ib[3] = q;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(1, 1);
	contr.contract(3, 3);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_iplq_pkjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{iplq} b_{pkjq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_iplq_pkjq(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
	index<4> ib1, ib2;
	ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
		ib[0] = p; ib[1] = k; ib[2] = j; ib[3] = q;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(1, 0);
	contr.contract(3, 3);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_iplq_pkqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{iplq} b_{pkqj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_iplq_pkqj(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
	index<4> ib1, ib2;
	ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = l; ia[3] = q;
		ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(1, 0);
	contr.contract(3, 2);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_ipql_kpqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{ipql} b_{kpqj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ipql_kpqj(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
	index<4> ib1, ib2;
	ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = q; ia[3] = l;
		ib[0] = k; ib[1] = p; ib[2] = q; ib[3] = j;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(1, 1);
	contr.contract(2, 2);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_ipql_pkqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{ipql} b_{pkqj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ipql_pkqj(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = ni - 1; ia2[1] = np - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
	index<4> ib1, ib2;
	ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = i; ia[1] = p; ia[2] = q; ia[3] = l;
		ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(1, 0);
	contr.contract(2, 2);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_pilq_kpjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{pilq} b_{kpjq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_pilq_kpjq(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
	index<4> ib1, ib2;
	ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = p; ia[1] = i; ia[2] = l; ia[3] = q;
		ib[0] = k; ib[1] = p; ib[2] = j; ib[3] = q;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(0, 1);
	contr.contract(3, 3);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_pilq_pkjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{pilq} b_{pkjq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_pilq_pkjq(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nl - 1; ia2[3] = nq - 1;
	index<4> ib1, ib2;
	ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nj - 1; ib2[3] = nq - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = p; ia[1] = i; ia[2] = l; ia[3] = q;
		ib[0] = p; ib[1] = k; ib[2] = j; ib[3] = q;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(0, 0);
	contr.contract(3, 3);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_piql_kpqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{piql} b_{kpqj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_piql_kpqj(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
	index<4> ib1, ib2;
	ib2[0] = nk - 1; ib2[1] = np - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = p; ia[1] = i; ia[2] = q; ia[3] = l;
		ib[0] = k; ib[1] = p; ib[2] = q; ib[3] = j;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(0, 1);
	contr.contract(2, 2);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_piql_pkqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq, double d)
	throw(libtest::test_exception) {

	//	c_{ijkl} = \sum_{pq} a_{piql} b_{pkqj}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_piql_pkqj(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << nq
		<< ", " << d << ")";
	std::string tns = tnss.str();

	try {

	index<4> ia1, ia2;
	ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nq - 1; ia2[3] = nl - 1;
	index<4> ib1, ib2;
	ib2[0] = np - 1; ib2[1] = nk - 1; ib2[2] = nq - 1; ib2[3] = nj - 1;
	index<4> ic1, ic2;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<4> dima(index_range<4>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//	Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();
	if(d == 0.0) for(size_t i = 0; i < szc; i++) dtc2[i] = 0.0;
	else for(size_t i = 0; i < szc; i++) dtc2[i] = dtc1[i];

	//	Generate reference data

	index<4> ia; index<4> ib; index<4> ic;
	double d1 = (d == 0.0) ? 1.0 : d;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {
		ia[0] = p; ia[1] = i; ia[2] = q; ia[3] = l;
		ib[0] = p; ib[1] = k; ib[2] = q; ib[3] = j;
		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aa(ia, dima), ab(ib, dimb), ac(ic, dimc);
		dtc2[ac.get_abs_index()] += d1 *
			dta[aa.get_abs_index()] * dtb[ab.get_abs_index()];
	}
	}
	}
	}
	}
	}
	for(size_t i = 0; i < szc; i++)
		if(fabs(dtc2[i]) > cij_max) cij_max = fabs(dtc2[i]) ;

	tca.ret_dataptr(dta); dta = 0; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = 0; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = 0;
	tcc_ref.ret_dataptr(dtc2); dtc2 = 0; tc_ref.set_immutable();
	}

	//	Invoke the contraction routine

	contraction2<2, 2, 2> contr(permutation<4>().permute(1, 3));
	contr.contract(0, 0);
	contr.contract(2, 2);
	if(d == 0.0) tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc);
	else tod_contract2<2, 2, 2>(contr, ta, tb).perform(tc, d);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max * k_thresh);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_ipqr_jpqr(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << nr << ")";
	std::string tns = tnss.str();

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
			ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_ipqr_jpqr_a(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
	std::string tns = tnss.str();

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
			ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ij_jpqr_iprq(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{jpqr} b_{iprq}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_jpqr_iprq(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
	std::string tns = tnss.str();

	index<4> ia1, ia2; ia2[0]=nj-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=np-1; ib2[2]=nr-1; ib2[3]=nq-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=j; ia[1]=p; ia[2]=q; ia[3]=r;
			ib[0]=i; ib[1]=p; ib[2]=r; ib[3]=q;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		if(d == 0.0) dtc2[dimc.abs_index(ic)] = cij;
		else dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	//~ contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
	contraction2<1, 1, 3> contr;
	contr.contract(1, 1);
	contr.contract(2, 3);
	contr.contract(3, 2);

	//~ tod_contract2<1, 1, 3> op(contr, ta, tb);
	tod_contract2<1, 1, 3> op(contr, tb, ta);
	if(d == 0.0) op.perform(tc);
	else op.perform(tc, d);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ij_pqir_pqjr(size_t ni, size_t nj,
	size_t np, size_t nq, size_t nr) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pqir_pqjr(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << nr << ")";
	std::string tns = tnss.str();

	index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(0, 0);
	contr.contract(1, 1);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}

	std::stringstream tnss;
	tnss << "tod_contract2_test::test_ij_pqir_pqjr_a(" << ni << ", " << nj
		<< ", " << np << ", " << nq << ", " << nr << ", " << d << ")";
	std::string tns = tnss.str();

	index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<4, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<2, double> tcc(tc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(0, 0);
	contr.contract(1, 1);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ijkl_pi_jklp(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

	//
	//	c_{ijkl} = \sum_p a_{pi} b_{jklp}
	//

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_pi_jklp(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ")";

	try {

	index<2> ia1, ia2;
	index<4> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = np - 1; ia2[1] = ni - 1;
	ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1; ib2[3] = np - 1;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cijkl_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//
	//	Fill in random input
	//

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

	//
	//	Generate reference data
	//

	index<2> ia; index<4> ib, ic;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {

		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aic(ic, dimc);
		double cijkl = 0.0;
		for(size_t p = 0; p < np; p++) {
			ia[0] = p; ia[1] = i;
			ib[0] = j; ib[1] = k; ib[2] = l; ib[3] = p;
			abs_index<2> aia(ia, dima);
			abs_index<4> aib(ib, dimb);
			cijkl += dta[aia.get_abs_index()]*
				dtb[aib.get_abs_index()];
		}
		dtc2[aic.get_abs_index()] = cijkl;
		if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	//
	//	Invoke the contraction routine
	//

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	tod_contract2<1, 3, 1>(contr, ta, tb).perform(tc);

	//
	//	Compare against the reference
	//

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
		cijkl_max*k_thresh);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_pi_jklp_a(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, double d) throw(libtest::test_exception) {

	//
	//	c_{ijkl} = c_{ijkl} + d * \sum_p a_{pi} b_{jklp}
	//

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_pi_jklp_a(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << d << ")";

	try {

	index<2> ia1, ia2;
	index<4> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = np - 1; ia2[1] = ni - 1;
	ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1; ib2[3] = np - 1;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<4> dimb(index_range<4>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cijkl_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//
	//	Fill in random input
	//

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = dtc2[i] = drand48();

	//
	//	Generate reference data
	//

	index<2> ia; index<4> ib, ic;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {

		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		abs_index<4> aic(ic, dimc);
		double cijkl = 0.0;
		for(size_t p = 0; p < np; p++) {
			ia[0] = p; ia[1] = i;
			ib[0] = j; ib[1] = k; ib[2] = l; ib[3] = p;
			abs_index<2> aia(ia, dima);
			abs_index<4> aib(ib, dimb);
			cijkl += dta[aia.get_abs_index()]*
				dtb[aib.get_abs_index()];
		}
		dtc2[aic.get_abs_index()] += d*cijkl;
		if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	//
	//	Invoke the contraction routine
	//

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);

	tod_contract2<1, 3, 1>(contr, ta, tb).perform(tc, d);

	//
	//	Compare against the reference
	//

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
		cijkl_max*k_thresh);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_jikl_pi_jpkl(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

	// c_{jikl} = \sum_p a_{pi} b_{jpkl}

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_jikl_pi_jpkl(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ")";
	std::string tns = tnss.str();

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
	index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib, ic;
	for(size_t j=0; j<nj; j++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
		double cjikl = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=p; ia[1]=i;
			ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
			cjikl += dta[dima.abs_index(ia)]*
				dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] = cjikl;
		if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<4> permc; permc.permute(0, 1);
	contraction2<1, 3, 1> contr(permc);
	contr.contract(0, 1);

	tod_contract2<1, 3, 1> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_jikl_pi_jpkl_a(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, double d) throw(libtest::test_exception) {

	// c_{jikl} = c_{jikl} + d \sum_p a_{pi} b_{jpkl}

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_jikl_pi_jpkl_a(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
	index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<4, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<4, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib, ic;
	for(size_t j=0; j<nj; j++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
		double cjikl = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=p; ia[1]=i;
			ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
			cjikl += dta[dima.abs_index(ia)]*
				dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] += d*cjikl;
		if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<4> permc; permc.permute(0, 1);
	contraction2<1, 3, 1> contr(permc);
	contr.contract(0, 1);

	tod_contract2<1, 3, 1> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ijkl_ijp_klp(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

	// c_{ijkl} = \sum_{p} a_{ijp} b_{klp}

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ijp_klp(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ")";
	std::string tns = tnss.str();

	index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
	index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
	index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
	index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
	index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima);
	tensor<3, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<3, double> tca(ta);
	tensor_ctrl<3, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<3> ia, ib; index<4> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=i; ia[1]=j; ia[2]=p;
			ib[0]=k; ib[1]=l; ib[2]=p;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<4> permc;
	contraction2<2, 2, 1> contr(permc);
	contr.contract(2, 2);

	tod_contract2<2, 2, 1> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, double d) throw(libtest::test_exception) {

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ijp_klp_a(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ", " << np << ", " << d << ")";
	std::string tns = tnss.str();

	index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
	index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
	index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
	index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
	index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima);
	tensor<3, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cij_max = 0.0;

	{
	tensor_ctrl<3, double> tca(ta);
	tensor_ctrl<3, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<3> ia, ib; index<4> ic;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=i; ia[1]=j; ia[2]=p;
			ib[0]=k; ib[1]=l; ib[2]=p;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	// Invoke the contraction routine

	permutation<4> permc;
	contraction2<2, 2, 1> contr(permc);
	contr.contract(2, 2);

	tod_contract2<2, 2, 1> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, cij_max*k_thresh);
}


void tod_contract2_test::test_ijkl_ij_kl(size_t ni, size_t nj,
	size_t nk, size_t nl) throw(libtest::test_exception) {

	//
	//	c_{ijkl} = a_{ij} b_{kl}
	//

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ij_kl(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ")";

	try {

	index<2> ia1, ia2;
	index<2> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = ni - 1; ia2[1] = nj - 1;
	ib2[0] = nk - 1; ib2[1] = nl - 1;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cijkl_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//
	//	Fill in random input
	//

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

	//
	//	Generate reference data
	//

	index<2> ia, ib; index<4> ic;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {

		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		ia[0] = i; ia[1] = j;
		ib[0] = k; ib[1] = l;
		abs_index<2> aia(ia, dima);
		abs_index<2> aib(ib, dimb);
		abs_index<4> aic(ic, dimc);
		double cijkl = dta[aia.get_abs_index()]*
			dtb[aib.get_abs_index()];
		dtc2[aic.get_abs_index()] = cijkl;
		if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	//
	//	Invoke the contraction routine
	//

	contraction2<2, 2, 0> contr;

	tod_contract2<2, 2, 0>(contr, ta, tb).perform(tc);

	//
	//	Compare against the reference
	//

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
		cijkl_max*k_thresh);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


void tod_contract2_test::test_ijkl_ij_lk(size_t ni, size_t nj,
	size_t nk, size_t nl) throw(libtest::test_exception) {

	//
	//	c_{ijkl} = a_{ij} b_{lk}
	//

	std::ostringstream tnss;
	tnss << "tod_contract2_test::test_ijkl_ij_lk(" << ni << ", " << nj
		<< ", " << nk << ", " << nl << ")";

	try {

	index<2> ia1, ia2;
	index<2> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = ni - 1; ia2[1] = nj - 1;
	ib2[0] = nl - 1; ib2[1] = nk - 1;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;

	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimensions<4> dimc(index_range<4>(ic1, ic2));
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	double cijkl_max = 0.0;

	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	tensor_ctrl<4, double> tcc(tc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	//
	//	Fill in random input
	//

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();
	for(size_t i = 0; i < szc; i++) dtc1[i] = drand48();

	//
	//	Generate reference data
	//

	index<2> ia, ib; index<4> ic;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t l = 0; l < nl; l++) {

		ic[0] = i; ic[1] = j; ic[2] = k; ic[3] = l;
		ia[0] = i; ia[1] = j;
		ib[0] = l; ib[1] = k;
		abs_index<2> aia(ia, dima);
		abs_index<2> aib(ib, dimb);
		abs_index<4> aic(ic, dimc);
		double cijkl = dta[aia.get_abs_index()]*
			dtb[aib.get_abs_index()];
		dtc2[aic.get_abs_index()] = cijkl;
		if(fabs(cijkl) > cijkl_max) cijkl_max = fabs(cijkl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();
	}

	//
	//	Invoke the contraction routine
	//

	permutation<4> permc;
	permc.permute(2, 3);
	contraction2<2, 2, 0> contr(permc);

	tod_contract2<2, 2, 0>(contr, ta, tb).perform(tc);

	//
	//	Compare against the reference
	//

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref,
		cijkl_max*k_thresh);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
