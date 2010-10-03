#ifndef LIBTENSOR_TOD_SYMCONTRACT2_TEST_H
#define LIBTENSOR_TOD_SYMCONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_symcontract2 class

	\ingroup libtensor_tests
**/
class tod_symcontract2_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
private: 
	void test_ij_ip_jp(size_t, size_t ) throw(libtest::test_exception);
	void test_ijab_iapq_pbqj(size_t, size_t, size_t, size_t ) throw(libtest::test_exception);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SYMCONTRACT2_TEST_H

