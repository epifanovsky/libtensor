#ifndef LIBTENSOR_CONTRACTION2_LIST_BUILDER_TEST_H
#define LIBTENSOR_CONTRACTION2_LIST_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::contraction2_list_builder class

	\ingroup libtensor_tests
**/
class contraction2_list_builder_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_LIST_BUILDER_TEST_H
