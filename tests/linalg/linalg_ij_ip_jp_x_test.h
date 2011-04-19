#ifndef LIBTENSOR_LINALG_IJ_IP_JP_X_TEST_H
#define LIBTENSOR_LINALG_IJ_IP_JP_X_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::linalg class (ij_ip_jp_x)

	\ingroup libtensor_tests
 **/
class linalg_ij_ip_jp_x_test : public linalg_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ij_ip_jp_x(size_t ni, size_t nj, size_t np, size_t sia,
		size_t sic, size_t sjb) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_IJ_IP_JP_X_TEST_H
