#ifndef LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_TEST_H
#define LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::so_proj_down_impl_part class

	\ingroup libtensor_tests
 **/
class so_proj_down_impl_part_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2(bool sign) throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_TEST_H

