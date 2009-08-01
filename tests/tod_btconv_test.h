#ifndef LIBTENSOR_TOD_BTCONV_TEST_H
#define LIBTENSOR_TOD_BTCONV_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_btconv class

	\ingroup libtensor_tests
**/
class tod_btconv_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);

	template<size_t N>
	void compare_ref(const char *test, tensor_i<N, double> &t,
		tensor_i<N, double> &t_ref, double thresh)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_TEST_H
