#ifndef LIBTENSOR_LINALG_IJK_PIKQ_JPQ_X_TEST_H
#define LIBTENSOR_LINALG_IJK_PIKQ_JPQ_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (ijk_pikq_jpq_x)

	\ingroup libtensor_tests
 **/
class linalg_ijk_pikq_jpq_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijk_pikq_jpq_x(size_t ni, size_t nj, size_t nk,
		size_t np, size_t nq) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJK_PIKQ_JPQ_X_TEST_H
