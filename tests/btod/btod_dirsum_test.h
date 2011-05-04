#ifndef LIBTENSOR_BTOD_DIRSUM_TEST_H
#define LIBTENSOR_BTOD_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_dirsum class

	\ingroup libtensor_tests_btod
**/
class btod_dirsum_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	// c_{ij} = a_i + b_j
	void test_ij_i_j_1(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = a_i + a_j (with more than 1 block, checking for symmetry)
	void test_ij_i_j_2(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ij} = a_i - a_j (with more than 1 block, checking for symmetry)
	void test_ij_i_j_3(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ijk} = a_{ij} + b_k
	void test_ijk_ij_k_1(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ikjl} = a_{ij} + b_{kl}
	void test_ikjl_ij_kl_1(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block)
	void test_ikjl_ij_kl_2(bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{ikjl} = a_{ij} + b_{kl} (with more than 1 block and symmetry elements)
	void test_ikjl_ij_kl_3(bool part, bool label, bool rnd, double d = 0.0)
		throw(libtest::test_exception);

	// c_{iklj} = a_{ij} + a_{kl} (with more than 1 block and symmetry elements)
	void test_iklj_ij_kl_1(bool rnd, double d = 0.0)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIRSUM_TEST_H
