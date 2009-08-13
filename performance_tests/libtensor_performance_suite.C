#include <libtensor.h>
#include "libtensor_performance_suite.h"

namespace libtensor {

libtensor_performance_suite::libtensor_performance_suite() 
	: libtest::test_suite("libtensor_performance") 
{
//	add_test("tod_add_p1", m_utf_tod_add_p1);
//	add_test("tod_add_p2", m_utf_tod_add_p2);
//	add_test("tod_add_p3", m_utf_tod_add_p3);
	add_test("expression_p", m_utf_expression_p);
}

}

