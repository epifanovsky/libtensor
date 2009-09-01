#ifndef LIBTENSOR_PERFORMANCE_TEST_SUITE_H
#define LIBTENSOR_PERFORMANCE_TEST_SUITE_H

#include <libtest.h>
#include "performance_test_scenario_i.h"

namespace libtensor {

/**	\brief Performance test suite for the tensor library (libtensor)
  
 	\ingroup libtensor_performance_tests
**/
class performance_test_suite : public libtest::test_suite {
	size_t m_ntests;
protected:
	/** \brief adds tests of a performance test scenario to the suite
	  	\param name Name of performance test scenario
	  	\param comment Comment to performance test scenario
	  	\param pts  performance_test_scenario object 
	 **/
	void add_tests( const char* name, const char* comment,
			performance_test_scenario_i& pts );
public:
	//!	Creates the suite
	performance_test_suite( const char* name );
	virtual ~performance_test_suite() {}
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SUITE_H

