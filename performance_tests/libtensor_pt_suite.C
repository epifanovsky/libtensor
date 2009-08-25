#include <libtensor.h>
#include "libtensor_pt_suite.h"

namespace libtensor {


libtensor_pt_suite::libtensor_pt_suite() 
	: performance_test_suite("libtensor_performance_test") 
{
	add_tests("tod_add (small)",m_tod_add_ptsc1);
	add_tests("tod_add (medium)",m_tod_add_ptsc2);
	add_tests("tod_add (large)",m_tod_add_ptsc3);
	add_tests("tod_contract2 (small)",m_tod_contract2_ptsc1);
	add_tests("tod_contract2 (medium)",m_tod_contract2_ptsc2);
	add_tests("tod_contract2 (large)",m_tod_contract2_ptsc3);
	add_tests("tod_copy (small)",m_tod_copy_ptsc1);
	add_tests("tod_copy (medium)",m_tod_copy_ptsc2);
	add_tests("tod_copy (large)",m_tod_copy_ptsc3);
	add_tests("tod_dotprod (small)",m_tod_dotprod_ptsc1);
	add_tests("tod_dotprod (medium)",m_tod_dotprod_ptsc2);
	add_tests("tod_dotprod (large)",m_tod_dotprod_ptsc3);
	add_tests("expressions (normal,bs 16)",m_expression_tests_n16);
	add_tests("expressions (normal,bs 4)",m_expression_tests_n4);
	add_tests("expressions (normal,bs 2)",m_expression_tests_n2);
	add_tests("expressions (large,bs 32)",m_expression_tests_l32);
	add_tests("expressions (large,bs 16)",m_expression_tests_l16);
	add_tests("expressions (large,bs 4)",m_expression_tests_l4);
}

}

