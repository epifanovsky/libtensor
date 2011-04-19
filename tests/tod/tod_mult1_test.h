#ifndef LIBTENSOR_TOD_MULT1_TEST_H
#define LIBTENSOR_TOD_MULT1_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_mult1 class

	\ingroup libtensor_tests
**/
class tod_mult1_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_pq_pq_1(size_t ni, size_t nj,
			bool recip) throw(libtest::test_exception);
	void test_pq_pq_2(size_t ni, size_t nj,
			bool recip, double coeff) throw(libtest::test_exception);
	void test_pqrs_qrps(size_t ni, size_t nj, size_t nk, size_t nl,
			bool recip, double coeff) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT1_TEST_H
