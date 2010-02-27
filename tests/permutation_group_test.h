#ifndef LIBTENSOR_PERMUTATION_GROUP_TEST_H
#define LIBTENSOR_PERMUTATION_GROUP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::permutation_group class

	\ingroup libtensor_tests
 **/
class permutation_group_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_TEST_H