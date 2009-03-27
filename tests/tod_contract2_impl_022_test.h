#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_022_TEST_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_022_TEST_H

#include <libtest.h>
#include "tod_contract2_impl_022.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_contract2_impl_022 class

	\ingroup libtensor_tests
**/
class tod_contract2_impl_022_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ij_pq_ijpq(size_t ni, size_t nj, size_t np, size_t nq)
		throw(libtest::test_exception);
	void test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np, size_t nq,
		double d) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_022_TEST_H

