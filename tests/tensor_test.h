#ifndef __LIBTENSOR_TENSOR_TEST_H
#define __LIBTENSOR_TENSOR_TEST_H

#include <libtest.h>
#include "tensor.h"
#include "tensor_operation_base.h"

namespace libtensor {

/**	\brief Tests the libtensor::tensor class
**/
class tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	class test_op_1_int : public tensor_operation_base<int,permutation> {
	public:
		virtual void perform(tensor_i<int> &t) throw(exception);
	};

	//!	Tests the constructor
	void test_ctor() throw(libtest::test_exception);

	//!	Tests immutability
	void test_immutable() throw(libtest::test_exception);

	//!	Tests operations
	void test_operation() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_TENSOR_TEST_H

