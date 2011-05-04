#ifndef LIBTENSOR_BTOD_SCALE_TEST_H
#define LIBTENSOR_BTOD_SCALE_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_tensor_i.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_scale class

	\ingroup libtensor_tests_btod
**/
class btod_scale_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<size_t N>
	void test_generic(const char *testname,
		block_tensor_i<N, double> &bt, double c)
		throw(libtest::test_exception);

	void test_0() throw(libtest::test_exception);
	void test_i(size_t i) throw(libtest::test_exception);

	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SCALE_TEST_H
