#include "performance_test_suite.h"

#include <string>

namespace libtensor {

performance_test_suite::performance_test_suite( const char* name ) 
	: libtest::test_suite(name) 
{
}

void 
performance_test_suite::add_tests( const char* name, performance_test_scenario_i& pts ) 
{
	for ( size_t i=0; i<pts.number_of_tests(); i++ ) {
		std::string fullname(name);
		fullname+=", ";
		fullname+=pts.test_name(i);
		add_test(fullname.c_str(),pts.test(i));
	}
}

}

