#ifndef LIBTENSOR_PT_SUITE_H
#define LIBTENSOR_PT_SUITE_H

#include <libtest/libtest.h>
#include <libtensor/libtensor.h>
#include "performance_test_suite.h"

#include "tod_add_scenario.h"
#include "tod_copy_scenario.h"
#include "tod_contract2_scenario.h"
#include "tod_dotprod_scenario.h"
#include "expression_test_scenario.h"

#include "dimensions_data.h"
#include "bispace_data.h"

namespace libtensor {

/**
	\brief Performance test suite for the tensor library (libtensor)

	This suite runs the following performance test scenarios:
	\li libtensor::tod_add_scenario
	\li libtensor::tod_contract2_scenario
	\li libtensor::tod_copy_scenario
	\li libtensor::tod_dotprod_scenario
	\li libtensor::expression_test_scenario

	\ingroup libtensor_performance_tests
**/
class libtensor_pt_suite : public performance_test_suite {

	typedef dimensions_data<2,2,2,2> dim2_t;
	typedef dimensions_data<2,2,2,4> dim4_t;
	typedef dimensions_data<2,2,2,8> dim8_t;
	typedef dimensions_data<2,2,2,16> dim16_t;
	typedef dimensions_data<2,2,2,32> dim32_t;
	typedef dimensions_data<2,2,2,32> dim64_t;

	typedef arbitrary_blocks_data<32,64,2> normal2_t;
	typedef arbitrary_blocks_data<32,64,4> normal4_t;
	typedef arbitrary_blocks_data<32,64,8> normal8_t;
	typedef arbitrary_blocks_data<32,64,16> normal16_t;
	typedef arbitrary_blocks_data<32,256,4> large4_t;
	typedef arbitrary_blocks_data<32,256,8> large8_t;
	typedef arbitrary_blocks_data<32,256,16> large16_t;
	typedef arbitrary_blocks_data<32,256,16> large32_t;

	tod_add_scenario<1000000,4,dim8_t> m_tod_add_ptsc1;
	tod_add_scenario<60000,4,dim16_t> m_tod_add_ptsc2;
	tod_add_scenario<4000,4,dim32_t> m_tod_add_ptsc3;

	tod_contract2_scenario<10000,2,2,2,dim8_t> m_tod_contract2_ptsc1;
	tod_contract2_scenario<600,2,2,2,dim16_t> m_tod_contract2_ptsc2;
	tod_contract2_scenario<40,2,2,2,dim32_t> m_tod_contract2_ptsc3;

	tod_copy_scenario<1000000,4,dim8_t> m_tod_copy_ptsc1;
	tod_copy_scenario<60000,4,dim16_t> m_tod_copy_ptsc2;
	tod_copy_scenario<4000,4,dim32_t> m_tod_copy_ptsc3;

	tod_dotprod_scenario<1000000,4,dim8_t> m_tod_dotprod_ptsc1;
	tod_dotprod_scenario<60000,4,dim16_t> m_tod_dotprod_ptsc2;
	tod_dotprod_scenario<4000,4,dim32_t> m_tod_dotprod_ptsc3;

	expression_test_scenario<400,normal16_t> m_expression_tests_n16;
	expression_test_scenario<200,normal8_t> m_expression_tests_n8;
	expression_test_scenario<100,normal4_t> m_expression_tests_n4;
	expression_test_scenario<800,large32_t> m_expression_tests_l32;
	expression_test_scenario<400,large16_t> m_expression_tests_l16;
	expression_test_scenario<200,large8_t> m_expression_tests_l8;
	expression_test_scenario<100,large4_t> m_expression_tests_l4;
public:
	//!	Creates the suite
	libtensor_pt_suite();
	virtual ~libtensor_pt_suite() {}
};

} // namespace libtensor

#endif // LIBTENSOR_PT_SUITE_H

