#ifndef LIBTENSOR_TOD_CONTRACT2_TEST_H
#define LIBTENSOR_TOD_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_contract2 class

	\ingroup libtensor_tests
**/
class tod_contract2_test : public libtest::unit_test {
private:
	static const double k_thresh; //!< Threshold multiplier

public:
	virtual void perform() throw(libtest::test_exception);

private:
	// c = \sum_p a_p b_p
	void test_0_p_p(size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_i = \sum_p a_p b_{pi}
	void test_i_p_pi(size_t ni, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_i = \sum_p a_p b_{ip}
	void test_i_p_ip(size_t ni, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_i = \sum_p a_{pi} b_p
	void test_i_pi_p(size_t ni, size_t np, double d= 0.0)
		throw(libtest::test_exception);

	// c_i = \sum_p a_{ip} b_p
	void test_i_ip_p(size_t ni, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{pi} b_{pj}
	void test_ij_pi_pj(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{pi} b_{jp}
	void test_ij_pi_jp(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{ip} b_{pj}
	void test_ij_ip_pj(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{ip} b_{jp}
	void test_ij_ip_jp(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{pj} b_{pi}
	void test_ij_pj_pi(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{pj} b_{ip}
	void test_ij_pj_ip(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{jp} b_{ip}
	void test_ij_jp_ip(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_p a_{jp} b_{pi}
	void test_ij_jp_pi(size_t ni, size_t nj, size_t np, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_{pq} a_{pqi} b_{pjq}
	void test_ij_pqi_pjq(size_t ni, size_t nj, size_t np, size_t nq,
		double d = 0.0) throw(libtest::test_exception);

	// c_{ij} = \sum_{pq} a_{ipq} b_{jqp}
	void test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq,
		double d = 0.0) throw(libtest::test_exception);

	// c_{ij} = \sum_{pq} a_{jpq} b_{iqp}
	void test_ij_jpq_iqp(size_t ni, size_t nj, size_t np, size_t nq,
		double d = 0.0) throw(libtest::test_exception);

	// c_{ij} = \sum_{pq} a_{pq} b_{ijpq}
	void test_ij_pq_ijpq(size_t ni, size_t nj, size_t np, size_t nq)
		throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}
	void test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np, size_t nq,
		double d) throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{kpjq}
	void test_ijkl_iplq_kpjq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{pkjq}
	void test_ijkl_iplq_pkjq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{iplq} b_{pkqj}
	void test_ijkl_iplq_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{ipql} b_{kpqj}
	void test_ijkl_ipql_kpqj(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{ipql} b_{pkqj}
	void test_ijkl_ipql_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pilq} b_{kpjq}
	void test_ijkl_pilq_kpjq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pilq} b_{pkjq}
	void test_ijkl_pilq_pkjq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{piql} b_{kpqj}
	void test_ijkl_piql_kpqj(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{piql} b_{pkqj}
	void test_ijkl_piql_pkqj(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pkiq} b_{jplq}
	void test_ijkl_pkiq_jplq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{pq} a_{pkiq} b_{pjlq}
	void test_ijkl_pkiq_pjlq(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}
	void test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr) throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}
	void test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d) throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{jpqr} b_{iprq}
	void test_ij_jpqr_iprq(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d = 0.0) throw(libtest::test_exception);

	// c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}
	void test_ij_pqir_pqjr(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr) throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}
	void test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d) throw(libtest::test_exception);

	// c_{ijkl} = \sum_{p} a_{pi} b_{jklp}
	void test_ijkl_pi_jklp(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np) throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{pi} b_{jklp}
	void test_ijkl_pi_jklp_a(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, double d) throw(libtest::test_exception);

	// c_{jikl} = \sum_{p} a_{pi} b_{jpkl}
	void test_jikl_pi_jpkl(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np) throw(libtest::test_exception);

	// c_{jikl} = c_{jikl} + d \sum_{p} a_{pi} b_{jpkl}
	void test_jikl_pi_jpkl_a(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, double d) throw(libtest::test_exception);

	// c_{ijkl} = \sum_{p} a_{ijp} b_{klp}
	void test_ijkl_ijp_klp(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np) throw(libtest::test_exception);

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}
	void test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, double d) throw(libtest::test_exception);

	// c_{ijkl} = a_{ij} b_{kl}
	void test_ijkl_ij_kl(size_t ni, size_t nj, size_t nk, size_t nl)
		throw(libtest::test_exception);

	// c_{ijkl} = a_{ij} b_{lk}
	void test_ijkl_ij_lk(size_t ni, size_t nj, size_t nk, size_t nl)
		throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_TEST_H

