#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_221_TEST_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_221_TEST_H

#include <libtest.h>
#include "tod_contract2_impl_221.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_contract2_impl<2,2,1> class

	\ingroup libtensor_tests
**/
class tod_contract2_impl_221_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijkl_ijp_klp(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np) throw(libtest::test_exception);
	void test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t np, double d) throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_221_TEST_H

