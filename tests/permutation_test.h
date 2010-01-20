#ifndef LIBTENSOR_PERMUTATION_TEST_H
#define LIBTENSOR_PERMUTATION_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutation class

	\ingroup libtensor_tests
**/
class permutation_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ctor() throw(libtest::test_exception);
	void test_permute() throw(libtest::test_exception);
	void test_apply_mask_1() throw(libtest::test_exception);
	void test_apply_mask_2() throw(libtest::test_exception);
	void test_apply_mask_3() throw(libtest::test_exception);
	void test_invert() throw(libtest::test_exception);
	void test_apply() throw(libtest::test_exception);
	void test_print() throw(libtest::test_exception);
};

}

#endif // LIBTENSOR_PERMUTATION_TEST_H

