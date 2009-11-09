#ifndef LIBTENSOR_ASYM_CONTRACT_TEST_H
#define	LIBTENSOR_ASYM_CONTRACT_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::asym_contract function

	\ingroup libtensor_tests
**/
class asym_contract_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_tt_1() throw(libtest::test_exception);
	void test_ee_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_ASYM_CONTRACT_TEST_H
