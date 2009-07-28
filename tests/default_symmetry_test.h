#ifndef LIBTENSOR_DEFAULT_SYMMETRY_TEST_H
#define LIBTENSOR_DEFAULT_SYMMETRY_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::default_symmetry class

	\ingroup libtensor_tests
**/
class default_symmetry_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_iterator() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_TEST_H
