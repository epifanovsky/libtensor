#ifndef LIBTENSOR_SEQUENCE_TEST_H
#define LIBTENSOR_SEQUENCE_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests the libtensor::sequence class

	\ingroup libtensor_tests
 **/
class sequence_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SEQUENCE_TEST_H

