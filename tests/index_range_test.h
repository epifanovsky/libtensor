#ifndef __LIBTENSOR_INDEX_RANGE_TEST_H
#define __LIBTENSOR_INDEX_RANGE_TEST_H

#include <libtest.h>
#include "index_range.h"

namespace libtensor {

/**	\brief Tests the index_range class
**/
class index_range_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_INDEX_RANGE_TEST_H

