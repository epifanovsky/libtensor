#ifndef LIBTENSOR_MAPPED_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_MAPPED_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::mapped_block_tensor class

	\ingroup libtensor_tests
 **/
class mapped_block_tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_MAPPED_BLOCK_TENSOR_TEST_H
