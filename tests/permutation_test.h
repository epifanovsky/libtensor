#ifndef LIBTENSOR_PERMUTATION_TEST_H
#define LIBTENSOR_PERMUTATION_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutation class

	\ingroup libtensor_tests
**/
class permutation_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//! Tests the constructors
	void test_ctor() throw(libtest::test_exception);

	//! Tests the permute method
	void test_permute() throw(libtest::test_exception);

	//! Tests the invert method
	void test_invert() throw(libtest::test_exception);

	//! Tests exceptions in the apply method
	void test_apply() throw(libtest::test_exception);

	//! Tests operator<<
	void test_print() throw(libtest::test_exception);
};

}

#endif // LIBTENSOR_PERMUTATION_TEST_H

