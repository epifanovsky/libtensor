#ifndef LIBTENSOR_PT_SUITE_H
#define LIBTENSOR_PT_SUITE_H

#include <libtest.h>
#include <libtensor.h>
#include "performance_test_suite.h"
#include "tod_add_scenario.h"

namespace libtensor {


/**
	\brief Performance test suite for the tensor library (libtensor)

	This suite runs the following performance test scenarios:
	\li libtensor::tod_add_test
	
**/
class libtensor_pt_suite : public performance_test_suite {
	template<size_t N> 
	struct Small {
		dimensions<N> dims();
	};
	template<size_t N> 
	struct Medium {
		dimensions<N> dims();
	};
	template<size_t N> 
	struct Large {
		dimensions<N> dims();
	};

	tod_add_scenario<4,Small<4> > m_tod_add_ptsc1;
	tod_add_scenario<4,Medium<4> > m_tod_add_ptsc2;
	tod_add_scenario<4,Large<4> > m_tod_add_ptsc3;
public:
	//!	Creates the suite
	libtensor_pt_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_PT_SUITE_H

