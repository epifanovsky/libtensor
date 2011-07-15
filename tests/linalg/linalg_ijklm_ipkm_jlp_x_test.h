#ifndef LIBTENSOR_LINALG_IJKLM_IPKM_JLP_X_TEST_H
#define LIBTENSOR_LINALG_IJKLM_IPKM_JLP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (ijklm_ipkm_jlp_x)

	\ingroup libtensor_tests_linalg
 **/
class linalg_ijklm_ipkm_jlp_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijklm_ipkm_jlp_x(size_t ni, size_t nj, size_t nk, size_t nl,
		size_t nm, size_t np) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJKLM_IPKM_JLP_X_TEST_H
