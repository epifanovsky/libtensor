#ifndef LIBTENSOR_SO_STABILIZE_IMPL_LABEL_TEST_H
#define LIBTENSOR_SO_STABILIZE_IMPL_LABEL_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::so_stabilize_impl_label class

	\ingroup libtensor_tests_sym
 **/
class so_stabilize_impl_label_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	static const char *k_table_id;
	void test_1a() throw(libtest::test_exception);
	void test_1b() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_STABILIZE_IMPL_LABEL_TEST_H

