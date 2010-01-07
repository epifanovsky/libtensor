#ifndef LIBTENSOR_DIRECT_PRODUCT_TEST_H
#define LIBTENSOR_DIRECT_PRODUCT_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the direct product expression

	\ingroup libtensor_tests
**/
class direct_product_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_label_1() throw(libtest::test_exception);
	void test_tt_1() throw(libtest::test_exception);
	void test_tt_2() throw(libtest::test_exception);
	void test_te_1() throw(libtest::test_exception);
	void test_et_1() throw(libtest::test_exception);
	void test_ee_1() throw(libtest::test_exception);
	void test_ee_2() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_DIRECT_PRODUCT_TEST_H
