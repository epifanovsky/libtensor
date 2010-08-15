#ifndef LIBTENSOR_LINALG_TEST_H
#define LIBTENSOR_LINALG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::linalg class

	\ingroup libtensor_tests
 **/
class linalg_test : public libtest::unit_test {
private:
	static const double k_thresh = 5e-14;

public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_x_p_p(size_t np, size_t spa, size_t spb)
		throw(libtest::test_exception);

	void test_i_i_x(size_t ni, size_t sia, size_t sic)
		throw(libtest::test_exception);

	void test_i_ip_p(size_t ni, size_t np, size_t sia, size_t sic,
		size_t spb) throw(libtest::test_exception);

	void test_i_pi_p(size_t ni, size_t np, size_t sic, size_t spa,
		size_t spb) throw(libtest::test_exception);

	void test_ij_i_j(size_t ni, size_t nj, size_t sia, size_t sic,
		size_t sjb) throw(libtest::test_exception);

	void test_ij_ip_jp(size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t sjb) throw(libtest::test_exception);

	void test_ij_ip_pj(size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t spb) throw(libtest::test_exception);

	void test_ij_pi_jp(size_t ni, size_t nj, size_t np, size_t sic,
		size_t sjb, size_t spa) throw(libtest::test_exception);

	void test_ij_pi_pj(size_t ni, size_t nj, size_t np, size_t sic,
		size_t spa, size_t spb) throw(libtest::test_exception);

	void test_x_pq_qp(size_t np, size_t nq, size_t spa, size_t sqb)
		throw(libtest::test_exception);

	void test_i_ipq_qp(size_t ni, size_t np, size_t nq, size_t sia,
		size_t sic, size_t spa, size_t sqb)
		throw(libtest::test_exception);

	void test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq,
		size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
		throw(libtest::test_exception);

	bool cmp(double diff, double ref);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_TEST_H
