#ifndef LIBTENSOR_EXPR_BINARY_TEST_H
#define LIBTENSOR_EXPR_BINARY_TEST_H

#include <libtest.h>
#include "expr_binary.h"

namespace libtensor {

/**	\brief Tests the libtensor::expr_binary class

	\ingroup libtensor_tests
**/
class expr_binary_test : public libtest::unit_test {
private:
	template<typename T1, typename T2>
	class add_op {
		static inline T1 eval(T1 a, T2 b) {
			return a+b;
		}
	};
public:
	virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_BINARY_TEST_H

