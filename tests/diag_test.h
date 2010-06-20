#ifndef LIBTENSOR_DIAG_TEST_H
#define	LIBTENSOR_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::diag function

	\ingroup libtensor_tests
 **/
class diag_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_t_1() throw(libtest::test_exception);
	void test_t_2() throw(libtest::test_exception);
	void test_t_3() throw(libtest::test_exception);
	void test_t_4() throw(libtest::test_exception);
	void test_e_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_DIAG_TEST_H
