#ifndef TOD_CONTRACT_SCENARIO_H
#define TOD_CONTRACT_SCENARIO_H

#include <libtest.h>
#include "tod_contract2_performance.h"
#include "performance_test_scenario_i.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Performance test scenario for the libtensor::tod_add class
 
 	\param N dimensions of the tensors to be added
 	\param X size of the tensors 

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats,size_t N, size_t M, size_t K, typename X>  
class tod_contract2_scenario
	: public performance_test_scenario_i 
{	
	unit_test_factory<tod_contract2_ref<Repeats,N,M,K,X> > m_ref;
	unit_test_factory<tod_contract2_p1<Repeats,N,M,K,X> > m_pt1;
	unit_test_factory<tod_contract2_p2<Repeats,N,M,K,X> > m_pt2;
	unit_test_factory<tod_contract2_p3<Repeats,N,M,K,X> > m_pt3;
	unit_test_factory<tod_contract2_p4<Repeats,N,M,K,X> > m_pt4;
 
public:
	tod_contract2_scenario();
};


template<size_t Repeats,size_t N, size_t M, size_t K, typename X>  
tod_contract2_scenario<Repeats,N,M,K,X>::tod_contract2_scenario()  
{
	add_test("reference",m_ref);
	add_test("test 1",m_pt1);
	add_test("test 2",m_pt2);
	add_test("test 3",m_pt3);
	add_test("test 4",m_pt4);
}


} // namespace libtensor

#endif // TOD_ADD_SCENARIO_H

