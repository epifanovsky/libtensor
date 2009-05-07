#ifndef LIBTENSOR_CONTRACTION2_TEST_H
#define	LIBTENSOR_CONTRACTION2_TEST_H

#include <libtest.h>
#include "contraction2.h"

namespace libtensor {

/**	\brief Tests the libtensor::contraction2 class

	\ingroup libtensor_tests
**/
class contraction2_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_TEST_H

