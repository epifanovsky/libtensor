#ifndef LIBTENSOR_LINALG_TEST_H
#define LIBTENSOR_LINALG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::linalg class

	\ingroup libtensor_tests
 **/
class linalg_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijkl_iplq_kpjq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_iplq_pkjq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_iplq_pkqj_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_ipql_pkqj_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pilq_kpjq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pilq_pkjq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_piql_kpqj_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_piql_pkqj_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pkiq_jplq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pkiq_jpql_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pkiq_pjlq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pkiq_pjql_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pliq_jpkq_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pliq_jpqk_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	void test_ijkl_pliq_pjqk_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, size_t nq) throw(libtest::test_exception);

	bool cmp(double diff, double ref);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_TEST_H
