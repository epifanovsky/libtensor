#ifndef LIBTENSOR_CONTRACTION2_PROCESSOR_TEST_H
#define LIBTENSOR_CONTRACTION2_PROCESSOR_TEST_H

#include <libtest.h>
#include "contraction2_processor.h"

namespace libtensor {

/**	\brief Tests the libtensor::contraction2_processor class

	\ingroup libtensor_tests
**/
class contraction2_processor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests c = \sum_p a_p b_p
	void test_0_p_p_p(size_t np) throw(libtest::test_exception);

	//!	Tests c_i = \sum_p a_{ip} b_p
	void test_i_p_ip_p(size_t ni, size_t np) throw(libtest::test_exception);

	//!	Tests c_i = \sum_p a_p b_{ip}
	void test_i_p_p_ip(size_t ni, size_t np) throw(libtest::test_exception);

	//!	Tests c_{ij} = \sum_p a_{ip} b_{jp}
	void test_ij_p_ip_jp(size_t ni, size_t nj, size_t np)
		throw(libtest::test_exception);

	//!	Tests c_{ji} = \sum_p a_{ip} b_{jp}
	void test_ji_p_ip_jp(size_t ni, size_t nj, size_t np)
		throw(libtest::test_exception);

	//!	Tests c_{kij} = \sum_p a_{ip} b_{jkp}
	void test_kij_p_ip_jkp(size_t ni, size_t nj, size_t nk, size_t np)
		throw(libtest::test_exception);

	//!	Tests c_{kji} = \sum_p a_{ikp} b_{jp}
	void test_kji_p_ikp_jp(size_t ni, size_t nj, size_t nk, size_t np)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_PROCESSOR_TEST_H
