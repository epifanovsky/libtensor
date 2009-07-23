#ifndef LIBTENSOR_PERM_SYMMETRY_TEST_H
#define LIBTENSOR_PERM_SYMMETRY_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::perm_symmetry class

	\ingroup libtensor_tests
**/
class perm_symmetry_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_is_same() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_PERM_SYMMETRY_TEST_H

