#ifndef LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_TEST_H
#define LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::block_index_subspace_builder class

	\ingroup libtensor_tests
 **/
class block_index_subspace_builder_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_0() throw(libtest::test_exception);
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_TEST_H
