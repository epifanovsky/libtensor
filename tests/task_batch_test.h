#ifndef LIBTENSOR_TASK_BATCH_TEST_H
#define LIBTENSOR_TASK_BATCH_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::task_batch class

	\ingroup libtensor_tests
**/
class task_batch_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void run_all_tests() throw(libtest::test_exception);
	void test_1() throw(libtest::test_exception);
	void test_2() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TASK_BATCH_TEST_H

