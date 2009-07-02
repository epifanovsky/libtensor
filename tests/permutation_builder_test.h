#ifndef LIBTENSOR_PERMUTATION_BUILDER_TEST_H
#define LIBTENSOR_PERMUTATION_BUILDER_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutation_builder class

	\ingroup libtensor_tests
**/
class permutation_builder_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

}

#endif // LIBTENSOR_PERMUTATION_BUILDER_TEST_H
