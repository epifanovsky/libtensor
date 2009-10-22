#ifndef LIBTENSOR_SYM_CONTRACT_TEST_H
#define	LIBTENSOR_SYM_CONTRACT_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::sym_contract function

	\ingroup libtensor_tests
**/
class sym_contract_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ee_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SYM_CONTRACT_TEST_H
