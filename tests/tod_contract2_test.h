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
	// c_{ij} = \sum_{pq} a_{pq} b_{ijpq}
	void test_ij_pq_ijpq(size_t ni, size_t nj, size_t np, size_t nq)
		throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}
	void test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np, size_t nq,
		double d) throw(libtest::test_exception);

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}
	void test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr) throw(libtest::test_exception);

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}
	void test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d) throw(libtest::test_exception);

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

