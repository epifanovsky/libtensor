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
	void test_ctor_1() throw(libtest::test_exception);

	void test_split_1() throw(libtest::test_exception);
	void test_split_2() throw(libtest::test_exception);
	void test_split_3() throw(libtest::test_exception);
	void test_split_4() throw(libtest::test_exception);

	void test_equals_1() throw(libtest::test_exception);
	void test_equals_2() throw(libtest::test_exception);
	void test_equals_3() throw(libtest::test_exception);
	void test_equals_4() throw(libtest::test_exception);
	void test_equals_5() throw(libtest::test_exception);

	void test_match_1() throw(libtest::test_exception);
	void test_match_2() throw(libtest::test_exception);
	void test_match_3() throw(libtest::test_exception);
	void test_match_4() throw(libtest::test_exception);
	void test_match_5() throw(libtest::test_exception);

	void test_permute_1() throw(libtest::test_exception);

	void test_exc_1() throw(libtest::test_exception);
	void test_exc_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_TEST_H
