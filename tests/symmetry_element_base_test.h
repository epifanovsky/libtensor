#ifndef LIBTENSOR_SYMMETRY_ELEMENT_BASE_TEST_H
#define LIBTENSOR_SYMMETRY_ELEMENT_BASE_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::symmetry_element_base class

	\ingroup libtensor_tests
**/
class symmetry_element_base_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_BASE_TEST_H
