#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_113_TEST_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_113_TEST_H

#include <libtest.h>
#include "tod_contract2_impl_113.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_contract2_impl<1,1,3> class

	\ingroup libtensor_tests
**/
class tod_contract2_impl_113_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr) throw(libtest::test_exception);
	void test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d) throw(libtest::test_exception);

	void test_ij_pqir_pqjr(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr) throw(libtest::test_exception);
	void test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np, size_t nq,
		size_t nr, double d) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_113_TEST_H

