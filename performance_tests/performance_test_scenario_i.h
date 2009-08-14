#ifndef PERFORMANCE_TEST_SCENARIO_I_H_
#define PERFORMANCE_TEST_SCENARIO_I_H_

#include <libtest.h>

namespace libtensor {
/** 
 	\brief Performance test scenario interface 
 	
 	A performance scenario is supposed to store multiple tests which should be
 	executed one after the other.  
 	  
 **/
class performance_test_scenario_i {
public: 
	//! returns number of tests in the scenario
	virtual size_t number_of_tests() = 0;
	//! returns name of the scenario
	virtual const char* test_name( size_t ) = 0;
	//! returns unit test factory for i-th test		
	virtual libtest::unit_test_factory_i& test( size_t ) = 0;		
};


}
#endif /*PERFORMANCE_TEST_SCENARIO_I_H_*/
