#ifndef LIBTENSOR_EWMULT_TEST_H
#define	LIBTENSOR_EWMULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::ewmult function

	\ingroup libtensor_tests_iface
 **/
class ewmult_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_tt_1() throw(libtest::test_exception);
	void test_tt_2() throw(libtest::test_exception);
	void test_te_1() throw(libtest::test_exception);
	void test_et_1() throw(libtest::test_exception);
	void test_ee_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_EWMULT_TEST_H
