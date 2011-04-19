#ifndef LIBTENSOR_BLOCK_MAP_TEST_H
#define LIBTENSOR_BLOCK_MAP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::block_map class

	\ingroup libtensor_tests
 **/
class block_map_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_create() throw(libtest::test_exception);
	void test_immutable() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_TEST_H
