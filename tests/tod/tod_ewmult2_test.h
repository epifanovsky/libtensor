#ifndef LIBTENSOR_TOD_EWMULT2_TEST_H
#define LIBTENSOR_TOD_EWMULT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::tod_ewmult2 class

	\ingroup libtensor_tests
 **/
class tod_ewmult2_test : public libtest::unit_test {
private:
	static const double k_thresh; //!< Threshold multiplier

public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_i_i_i(size_t ni, double d = 0.0)
		throw(libtest::test_exception);
	void test_ij_ij_ij(size_t ni, size_t nj, double d = 0.0)
		throw(libtest::test_exception);
	void test_ij_ij_ji(size_t ni, size_t nj, double d = 0.0)
		throw(libtest::test_exception);
	void test_ijk_jki_kij(size_t ni, size_t nj, size_t nk, double d = 0.0)
		throw(libtest::test_exception);
	void test_ijk_ik_kj(size_t ni, size_t nj, size_t nk, double d = 0.0)
		throw(libtest::test_exception);
	void test_ijkl_kj_ikl(size_t ni, size_t nj, size_t nk, size_t nl,
		double d = 0.0) throw(libtest::test_exception);
	void test_ijkl_ljk_jil(size_t ni, size_t nj, size_t nk, size_t nl,
		double d = 0.0) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_EWMULT2_TEST_H
