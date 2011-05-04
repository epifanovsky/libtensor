#ifndef LIBTENSOR_LINALG_IJ_IPQ_JQP_X_TEST_H
#define LIBTENSOR_LINALG_IJ_IPQ_JQP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (ij_ipq_jqp_x)

	\ingroup libtensor_tests_linalg
 **/
class linalg_ij_ipq_jqp_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ij_ipq_jqp_x(size_t ni, size_t nj, size_t np, size_t nq,
		size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
		throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJ_IPQ_JQP_X_TEST_H
