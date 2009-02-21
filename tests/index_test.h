#ifndef __LIBTENSOR_INDEX_TEST_H
#define __LIBTENSOR_INDEX_TEST_H

#include <libtest.h>
#include "index.h"

namespace libtensor {

/**	\brief Tests the index class
**/
class index_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the constructors
	void test_ctor() throw(libtest::test_exception);

	//!	Tests the index::less() method
	void test_less() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // __LIBTENSOR_INDEX_TEST_H

