#ifndef EXPRESSION_TEST_SCENARIO_H
#define EXPRESSION_TEST_SCENARIO_H

#include <libtest.h>
#include "performance_test_scenario_i.h"
#include "expression_performance_test.h"
#include "test_expressions.h"

using libtest::unit_test_factory;

namespace libtensor {
	
/**	\brief Performance test scenario for various expression
 	
 	\tparam Repeats number of times each test is repeated
 	\tparam BiSpaceData sizes of the block tensors in the expression

	This performance test scenario includes expressions
	\li test_expression_add 

	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, typename BiSpaceData>  
class expression_test_scenario
	: public performance_test_scenario_i 
{	
	typedef expression_performance_test<Repeats,test_expression_add,BiSpaceData> add_expression_test;
	 
	unit_test_factory<add_expression_test> m_expression_add;
 	// add future expression here
public:
	expression_test_scenario();
};

template<size_t Repeats, typename BiSpaceData>  
expression_test_scenario<Repeats,BiSpaceData>::expression_test_scenario()
{
	add_test("add expression",m_expression_add);
}

	
	
} // namespace libtensor

#endif // EXPRESSION_TEST_SCENARIO_H
