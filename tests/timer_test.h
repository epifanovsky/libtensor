#ifndef LIBTENSOR_TIMER_TEST_H
#define LIBTENSOR_TIMER_TEST_H

#include <libtest.h>
#include <ctime>

namespace libtensor {

/**	\brief Tests the libtensor::timer class

	\ingroup libtensor_tests
**/
class timer_test : public libtest::unit_test {
	clock_t calc( double&, unsigned int ); 
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TIMER_TEST_H

