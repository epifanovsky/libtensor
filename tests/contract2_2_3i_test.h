#ifndef LIBTENSOR_CONTRACT2_2_3I_TEST_H
#define LIBTENSOR_CONTRACT2_2_3I_TEST_H

#include <libtest.h>
#include "contract2_2_3i.h"

namespace libtensor {

/**	\brief Tests the libtensor::contract2_2_3i class

	\ingroup libtensor_tests
**/
class contract2_2_3i_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ij_klm_klim_kljm(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t nm) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_3I_TEST_H

