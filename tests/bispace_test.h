#ifndef LIBTENSOR_BISPACE_TEST_H
#define	LIBTENSOR_BISPACE_TEST_H

#include <libtest.h>
#include "bispace.h"

namespace libtensor {

/**	\brief Tests the libtensor::bispace class

	\ingroup libtensor_tests
 **/
class bispace_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_TEST_H

