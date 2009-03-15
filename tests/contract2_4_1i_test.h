#ifndef LIBTENSOR_CONTRACT2_4_1I_TEST_H
#define LIBTENSOR_CONTRACT2_4_1I_TEST_H

#include <libtest.h>
#include "contract2_4_1i.h"

namespace libtensor {

/**	\brief Tests the libtensor::contract2_4_1i class

	\ingroup libtensor_tests
**/
class contract2_4_1i_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_jikl_m_mi_jmkl(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t nm) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_4_1I_TEST_H

