#ifndef LIBTENSOR_MP_SAFE_TENSOR_TEST_H
#define LIBTENSOR_MP_SAFE_TENSOR_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/mp/mp_safe_tensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::mp_safe_tensor class

	\ingroup libtensor_tests
**/
class mp_safe_tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	virtual void test_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_MP_SAFE_TENSOR_TEST_H
