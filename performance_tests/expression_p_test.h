#ifndef LIBTENSOR_EXPRESSION_P_TEST_H
#define LIBTENSOR_EXPRESSION_P_TEST_H

#include <libtest.h>

namespace libtensor {

/**	\brief Tests performance of the expression evaluation in libtensor

	\ingroup libtensor_tests
**/
class expression_p_test 
	: public libtest::unit_test, public timings<expression_p_test> 
{
	friend timings<tod_add_p1_test>;
	static const char* k_clazz; 
public:
	virtual void perform() throw(libtest::test_exception);
};

}

#endif // LIBTENSOR_PERMUTATION_TEST_H

