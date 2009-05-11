#ifndef LIBTENSOR_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_BLOCK_TENSOR_TEST_H

#include <libtest.h>
#include "block_tensor.h"

namespace libtensor {

/**	\brief Tests the libtensor::block_tensor class

	\ingroup libtensor_tests
 **/
class block_tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_TEST_H
