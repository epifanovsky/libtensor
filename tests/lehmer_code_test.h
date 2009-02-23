#ifndef LIBTENSOR_LEHMER_CODE_TEST_H
#define LIBTENSOR_LEHMER_CODE_TEST_H

#include <libtest.h>
#include "lehmer_code.h"

namespace libtensor {

/**	\brief Tests the libtensor::lehmer_code class

	\ingroup libtensor_tests
**/
class lehmer_code_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the code for a given %tensor order
	void test_code(const size_t order) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_LEHMER_CODE_TEST_H

