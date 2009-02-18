#ifndef __LIBTENSOR_TENSOR_TEST_H
#define __LIBTENSOR_TENSOR_TEST_H

#include <libtest.h>
#include "tensor.h"

namespace libtensor {

/**	\brief Tests the libtensor::tensor class
**/
class tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructor
	void test_ctor() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_TEST_H

