#ifndef LIBTENSOR_PARTITION_SET_TEST_H
#define LIBTENSOR_PARTITION_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::partition_set class

	\ingroup libtensor_tests
 **/
class partition_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4(bool sign) throw(libtest::test_exception);
	void test_5(bool sign) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_PARTITION_SET_TEST_H

