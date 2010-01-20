#ifndef LIBTENSOR_DOT_PRODUCT_TEST_H
#define	LIBTENSOR_DOT_PRODUCT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::dot_product function

	\ingroup libtensor_tests
**/
class dot_product_test : public libtest::unit_test {
private:
	static const double k_thresh;

public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(size_t ni) throw(libtest::test_exception);
	void test_2_ij_ij(size_t ni, size_t nj) throw(libtest::test_exception);
	void test_2_ij_ji(size_t ni, size_t nj) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_TEST_H
