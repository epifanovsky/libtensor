#ifndef LIBTENSOR_TOD_DIRSUM_TEST_H
#define LIBTENSOR_TOD_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_dirsum class

	\ingroup libtensor_tests
**/
class tod_dirsum_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	// c_{ij} = a_i + b_j
	void test_ij_i_j(size_t ni, size_t nj, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ikj} = a_{ij} + b_k
	void test_ikj_ij_k_1(size_t ni, size_t nj, size_t nk,
		double d = 0.0) throw(libtest::test_exception);

	// c_{ikjl} = a_{ij} + b_{kl}
	void test_ikjl_ij_kl_1(size_t ni, size_t nj, size_t nk, size_t nl,
		double d = 0.0) throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_DIRSUM_TEST_H

