#ifndef LIBTENSOR_LINALG_X_PQ_QP_TEST_H
#define LIBTENSOR_LINALG_X_PQ_QP_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (x_pq_qp)

	\ingroup libtensor_tests
 **/
class linalg_x_pq_qp_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_x_pq_qp(size_t np, size_t nq, size_t spa, size_t sqb)
		throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_X_PQ_QP_TEST_H
