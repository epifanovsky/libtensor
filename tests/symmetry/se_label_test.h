#ifndef LIBTENSOR_SE_LABEL_TEST_H
#define LIBTENSOR_SE_LABEL_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::se_label class

	\ingroup libtensor_tests_sym
 **/
class se_label_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	static const char *table_id;

	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6() throw(libtest::test_exception);
	void test_7() throw(libtest::test_exception);
	void test_8() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_H

