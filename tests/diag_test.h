#ifndef LIBTENSOR_DIAG_TEST_H
#define	LIBTENSOR_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::diag function

	\ingroup libtensor_tests
 **/
class diag_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:

};

} // namespace libtensor

#endif // LIBTENSOR_DIAG_TEST_H
