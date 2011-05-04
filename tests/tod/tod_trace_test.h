#ifndef LIBTENSOR_TOD_TRACE_TEST_H
#define LIBTENSOR_TOD_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_trace class

	\ingroup libtensor_tests_tod
 **/
class tod_trace_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(size_t ni) throw(libtest::test_exception);
	void test_2(size_t ni) throw(libtest::test_exception);
	void test_3(size_t ni, size_t nj) throw(libtest::test_exception);
	void test_4(size_t ni, size_t nj) throw(libtest::test_exception);
	void test_5(size_t ni, size_t nj, size_t nk)
		throw(libtest::test_exception);
	void test_6(size_t ni, size_t nj, size_t nk)
		throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_TRACE_TEST_H
