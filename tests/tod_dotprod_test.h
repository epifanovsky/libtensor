#ifndef LIBTENSOR_TOD_DOTPROD_TEST_H
#define LIBTENSOR_TOD_DOTPROD_TEST_H

#include <cmath>
#include <cstdlib>
#include <libtensor.h>
#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_dotprod class

	\ingroup libtensor_tests
**/
class tod_dotprod_test : public libtest::unit_test {
private:
	static const double k_thresh; //!< Threshold multiplier

public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(size_t ni) throw(libtest::test_exception);
	void test_2(size_t ni, size_t nj, const permutation<2> &perm)
		throw(libtest::test_exception);
	void test_4(size_t ni, size_t nj, size_t nk, size_t nl,
		const permutation<4> &perm) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_TEST_H

