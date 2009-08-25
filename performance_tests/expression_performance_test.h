#ifndef EXPRESSION_PERFORMANCE_TEST_H
#define EXPRESSION_PERFORMANCE_TEST_H

#include "performance_test.h"

namespace libtensor {

/**	\brief Expression performance tests
 	\tparam Repeats Number of repeats
 	\tparam Expr Expression object
 	\tparam BiSpaceData bispaces for various index types 
 	
 	Does the performance test of an expression given by Expr and block tensors
 	of sizes specified in BiSpaceData. For this to work properly derive Expr 
 	from test_expression_i and BiSpaceData from bispace_data_i.     

	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, typename Expr, typename BiSpaceData>
class expression_performance_test
	: public performance_test<Repeats>
{
protected:
	//! Implementation of libtensor::performance_test<Repeats>
	virtual void do_calculate();
};

template<size_t Repeats, typename Expr, typename BiSpaceData>
void expression_performance_test<Repeats,Expr,BiSpaceData>::do_calculate()
{
	BiSpaceData bisd;
	Expr expr;
	expr.initialize(bisd);
	expr.calculate();
}

} // namespace libtensor

#endif // EXPRESSION_PERFORMANCE_TEST_H

