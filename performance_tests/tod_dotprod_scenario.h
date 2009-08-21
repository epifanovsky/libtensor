#ifndef TOD_DOTPROD_SCENARIO_H
#define TOD_DOTPROD_SCENARIO_H

#include <libtest.h>
#include "tod_dotprod_performance.h"
#include "performance_test_scenario_i.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Performance test scenario for the libtensor::tod_dotprod class
 
 	\param N dimensions of the tensors to be multiplied
 	\param X size of the tensors 

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, typename X>  
class tod_dotprod_scenario
	: public performance_test_scenario_i 
{	
	unit_test_factory<tod_dotprod_ref<Repeats,X> > m_ref;
	unit_test_factory<tod_dotprod_p1<Repeats,N,X> > m_pt1;
	unit_test_factory<tod_dotprod_p2<Repeats,N,X> > m_pt2;
	unit_test_factory<tod_dotprod_p3<Repeats,N,X> > m_pt3;
 
public:
	tod_dotprod_scenario();
};


template<size_t Repeats, size_t N, typename X>  
tod_dotprod_scenario<Repeats,N,X>::tod_dotprod_scenario()  
{
	add_test("reference",m_ref);
	add_test("test 1",m_pt1); 
	add_test("test 2",m_pt2); 
	add_test("test 3",m_pt3); 
}


} // namespace libtensor

#endif // TOD_DOTPROD_SCENARIO_H

