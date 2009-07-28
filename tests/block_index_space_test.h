#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_TEST_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::block_index_space class

	\ingroup libtensor_tests
 **/
class block_index_space_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_exc_1() throw(libtest::test_exception);
	void test_exc_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_TEST_H
