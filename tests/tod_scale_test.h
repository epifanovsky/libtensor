#ifndef LIBTENSOR_TOD_SCALE_TEST_H
#define LIBTENSOR_TOD_SCALE_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_scale class

	\ingroup libtensor_tests
**/
class tod_scale_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<size_t N>
	void test_generic(const char *testname, const dimensions<N> &d,
		double c) throw(libtest::test_exception);

	void test_0() throw(libtest::test_exception);
	void test_i(size_t i) throw(libtest::test_exception);
	void test_ij(size_t i, size_t j) throw(libtest::test_exception);
	void test_ijkl(size_t i, size_t j, size_t k, size_t l)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SCALE_TEST_H

