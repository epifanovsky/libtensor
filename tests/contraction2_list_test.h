#ifndef LIBTENSOR_CONTRACTION2_LIST_TEST_H
#define	LIBTENSOR_CONTRACTION2_LIST_TEST_H

#include <libtest.h>
#include "contraction2_list.h"

namespace libtensor {

/**	\brief Tests the libtensor::contraction2_list class

	\ingroup libtensor_tests
**/
class contraction2_list_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor


#endif // LIBTENSOR_CONTRACTION2_LIST_TEST_H

