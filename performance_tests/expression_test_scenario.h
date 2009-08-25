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
	typedef expression_performance_test<Repeats,
		test_expression_simple_add,BiSpaceData> expression_test_simple_add;
	typedef expression_performance_test<Repeats,
		test_expression_permute_add,BiSpaceData> expression_test_permute_add;
	typedef expression_performance_test<Repeats,
		test_expression_simple_copy,BiSpaceData> expression_test_simple_copy;
	typedef expression_performance_test<Repeats,
		test_expression_permute_copy,BiSpaceData> expression_test_permute_copy;
	 
	unit_test_factory<expression_test_simple_add> m_expression_simple_add;
	unit_test_factory<expression_test_permute_add> m_expression_permute_add;
	unit_test_factory<expression_test_simple_copy> m_expression_simple_copy;
	unit_test_factory<expression_test_permute_copy> m_expression_permute_copy;
 	// add future expression here
public:
	expression_test_scenario();
	
	virtual ~expression_test_scenario() {}
};

template<size_t Repeats, typename BiSpaceData>  
expression_test_scenario<Repeats,BiSpaceData>::expression_test_scenario()
{
	add_test("simple add",m_expression_simple_add);
//	add_test("permute add",m_expression_permute_add);
//	add_test("simple copy",m_expression_simple_copy);
//	add_test("permute copy",m_expression_permute_copy);
}

	
	
} // namespace libtensor

#endif // EXPRESSION_TEST_SCENARIO_H
