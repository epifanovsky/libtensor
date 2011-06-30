#ifndef LIBTENSOR_LINALG_IJK_IPQ_KJQP_X_TEST_H
#define LIBTENSOR_LINALG_IJK_IPQ_KJQP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (ijk_ipq_kjqp_x)

	\ingroup libtensor_tests_linalg
 **/
class linalg_ijk_ipq_kjqp_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijk_ipq_kjqp_x(size_t ni, size_t nj, size_t nk,
		size_t np, size_t nq) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJK_IPQ_KJQP_X_TEST_H
