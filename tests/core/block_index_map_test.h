#ifndef LIBTENSOR_BLOCK_INDEX_MAP_TEST_H
#define LIBTENSOR_BLOCK_INDEX_MAP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::block_index_map class

	\ingroup libtensor_tests_core
 **/
class block_index_map_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_MAP_TEST_H
