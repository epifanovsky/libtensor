#ifndef TOD_ADD_SCENARIO_H
#define TOD_ADD_SCENARIO_H

#include <libtest.h>
#include "tod_add_performance.h"
#include "performance_test_scenario_i.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Performance test scenario for the libtensor::tod_add class
 
 	\param N dimensions of the tensors to be added
 	\param X which contains the information about the dimensions of the tensors 

	\ingroup libtensor_tests
**/
template<size_t N, typename X>  
class tod_add_scenario
	: public performance_test_scenario_i 
{	
	unit_test_factory<tod_add_ref<10,X> > m_ref;
	unit_test_factory<tod_add_p1<10,N,X> > m_pt1;
	unit_test_factory<tod_add_p2<10,N,X> > m_pt2;
	unit_test_factory<tod_add_p3<10,N,X> > m_pt3;
 
 	const char* m_name_ref;
 	const char* m_name_pt1;
 	const char* m_name_pt2;
 	const char* m_name_pt3;
public:
	tod_add_scenario();
	virtual size_t number_of_tests();
	virtual const char* test_name( size_t );
	virtual libtest::unit_test_factory_i& test( size_t );
};


template<size_t N, typename X>  
tod_add_scenario<N,X>::tod_add_scenario()  
	: m_name_ref("reference"), m_name_pt1("test 1"), 
		m_name_pt2("test 2"), m_name_pt3("test 3")
{
}

template<size_t N, typename X>  
size_t tod_add_scenario<N,X>::number_of_tests() 
{ return 4; }

template<size_t N, typename X> 
const char* tod_add_scenario<N,X>::test_name( size_t i ) 
{ 
	switch (i) {
		case 0: return m_name_ref;
		case 1: return m_name_pt1; 
		case 2: return m_name_pt2; 
		case 3: return m_name_pt3;
		default: 
			throw overflow("libtensor","tod_add_scenario<N,X>","test_name(size_t)",
							__FILE__,__LINE__,"Number of tests exceeded!"); 
	}
}
	
template<size_t N, typename X> 
libtest::unit_test_factory_i& tod_add_scenario<N,X>::test( size_t i ) 
{ 
	switch (i) {
		case 0: return m_ref;
		case 1: return m_pt1; 
		case 2: return m_pt2; 
		case 3: return m_pt3;
		default: 
			throw overflow("libtensor","tod_add_scenario<N,X>","test(size_t)",
							__FILE__,__LINE__,"Number of tests exceeded!"); 
	}
}

} // namespace libtensor

#endif // TOD_ADD_SCENARIO_H

