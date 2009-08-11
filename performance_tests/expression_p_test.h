#ifndef LIBTENSOR_PERMUTATION_TEST_H
#define LIBTENSOR_PERMUTATION_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutation class

	\ingroup libtensor_tests
**/
class expression_p_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

}

#endif // LIBTENSOR_PERMUTATION_TEST_H

