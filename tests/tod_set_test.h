#ifndef __LIBTENSOR_TOD_SET_TEST_H
#define __LIBTENSOR_TOD_SET_TEST_H

#include <libtest.h>
#include "tod_set.h"

namespace libtensor {

/**	\brief Tests the tod_set class
**/
class tod_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_TOD_SET_TEST_H

