#ifndef LIBTENSOR_BTOD_DIRSUM_TEST_H
#define LIBTENSOR_BTOD_DIRSUM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_dirsum class

	\ingroup libtensor_tests
**/
class btod_dirsum_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	// c_{ij} = a_i + b_j
	void test_ij_i_j_1(double d = 0.0) throw(libtest::test_exception);

	// c_{ikjl} = a_{ij} + b_{kl}
	void test_ikjl_ij_kl_1(double d = 0.0) throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIRSUM_TEST_H