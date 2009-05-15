#ifndef LIBTENSOR_PERMUTATOR_TEST_H
#define LIBTENSOR_PERMUTATOR_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutator class

	\ingroup libtensor_tests
**/
class permutator_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_PERMUTATOR_TEST_H

