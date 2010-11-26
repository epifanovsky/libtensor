#ifndef LIBTENSOR_TRANSF_LIST_TEST_H
#define LIBTENSOR_TRANSF_LIST_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::transf_list class

	\ingroup libtensor_tests
 **/
class transf_list_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5a() throw(libtest::test_exception);
	void test_5b() throw(libtest::test_exception);
	void test_5c() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TRANSF_LIST_TEST_H
