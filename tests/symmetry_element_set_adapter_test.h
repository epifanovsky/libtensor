#ifndef LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H
#define LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::symmetry_element_set_adapter class

	\ingroup libtensor_tests
 **/
class symmetry_element_set_adapter_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H
