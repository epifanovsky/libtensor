#ifndef LIBTENSOR_BLOCK_TENSOR_TEST_H
#define LIBTENSOR_BLOCK_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::block_tensor class

	\ingroup libtensor_tests
 **/
class block_tensor_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_req_aux_block_1() throw(libtest::test_exception);
	void test_orbits_1() throw(libtest::test_exception);
	void test_orbits_2() throw(libtest::test_exception);
	void test_orbits_3() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_TEST_H
