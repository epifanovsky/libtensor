#ifndef LIBTENSOR_LINALG_I_IPQ_QP_X_TEST_H
#define LIBTENSOR_LINALG_I_IPQ_QP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (i_ipq_qp_x)

	\ingroup libtensor_tests_linalg
 **/
class linalg_i_ipq_qp_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_i_ipq_qp_x(size_t ni, size_t np, size_t nq, size_t sia,
		size_t sic, size_t spa, size_t sqb)
		throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_I_IPQ_QP_X_TEST_H
