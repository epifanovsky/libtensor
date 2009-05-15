#ifndef LIBTENSOR_DEFAULT_SYMMETRY_TEST_H
#define LIBTENSOR_DEFAULT_SYMMETRY_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the default_symmetry class
**/
class default_symmetry_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_TEST_H

