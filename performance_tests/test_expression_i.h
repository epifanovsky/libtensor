#ifndef TEST_EXPRESSION_I_H
#define TEST_EXPRESSION_I_H

#include "bispace_data.h"

namespace libtensor {
	
	
/** \brief Base class for representing an expression
  
	Base class for expression evaluation in expression performance tests. 
	Each derived class must implement the two functions:
	\li initialize( const bispace_data_i<N>& bispaces ) which initializes all 
		necessary block tensors required in the calculation based on the 
		information in bispaces	   
	\li calculate() performs the evaluation and calculation of the expression.
	
	\ingroup libtensor_performance_tests
 **/
class test_expression_i
{
public:
	virtual void initialize( const bispace_data_i& bispaces ) = 0;
	virtual void calculate() = 0;
}; 

} // namespace libtensor

#endif // TEST_EXPRESSION_I_H 
