#ifndef __LIBTENSOR_PERMUTATOR_TEST_H
#define __LIBTENSOR_PERMUTATOR_TEST_H

#include <libtest.h>
#include "permutator.h"

namespace libtensor {

/**	\brief Tests the permutator class
**/
class permutator_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_PERMUTATOR_TEST_H

