#ifndef LIBTENSOR_SO_UNION_IMPL_PERM_TEST_H
#define LIBTENSOR_SO_UNION_IMPL_PERM_TEST_H

#include <libtest.h>

namespace libtensor {


/**	\brief Tests the libtensor::so_union_impl_perm class

	\ingroup libtensor_tests
**/
class so_union_impl_perm_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_UNION_IMPL_PERM_TEST_H

