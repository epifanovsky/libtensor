#ifndef LIBTENSOR_TIMINGS  
	#define LIBTENSOR_TIMINGS   
	#define SET_LIBTENSOR_TIMINGS
#endif 

#include <libtensor.h>
#include "timings_test.h"
#include "../timings.h"
#include "../global_timings.h"

namespace libtensor {

namespace timings_test_ns {


class timed_class : public timings<timed_class>  {
public:
	static const char* k_clazz;
	double some_function() {
		start_timer();
		double res=0.0;
		for (unsigned int i=0; i<10; i++) 
			res+=i/10000.0;
		stop_timer();
		
		return res;
	}
};

const char *timed_class::k_clazz="timed_class";

class timed_class2 : public timings<timed_class2>  {
public:
	static const char* k_clazz;
	double some_function() {
		start_timer("some_function()");
		double res=0.0;
		for (unsigned int i=0; i<10; i++) {
			res+=i/1000.0;
		}
		stop_timer("some_function()");
		
		return res; 
	}
	
	double some_other_function() {
		start_timer("some_other_function()");
		double res=0.0;
		for (unsigned int i=0; i<10; i++) 
			res+=i/1000.0;
		stop_timer("some_other_function()");
		
		return res;
	}
	
	double wrong_function() {
		start_timer("wrong_function()");
		double res=0.0;
		for (unsigned int i=0; i<10; i++) 
			res+=i/10.0;
		stop_timer("rong_function()");
		
		return res;
	}
	
};

const char *timed_class2::k_clazz="timed_class2";
	
} // namespace timings_test_ns

void timings_test::perform() throw(libtest::test_exception) {
	global_timings::get_instance().reset();
	
	timings_test_ns::timed_class tc1, tc2;
	
	try {
		tc1.some_function();
		tc1.some_function();
		tc2.some_function();
		global_timings::get_instance().get_time("timed_class");
		
		
	} catch ( exception& e ) {
		fail_test("timings_test::perform()",__FILE__,__LINE__,e.what());
	}
	
	timings_test_ns::timed_class2 tc3;
	
	try {
		tc3.some_function();
		tc3.some_other_function();
		tc3.some_function();
		
		global_timings::get_instance().get_time("timed_class2::some_function()");
		global_timings::get_instance().get_time("timed_class2::some_other_function()");		
	} catch ( exception& e ) {
		fail_test("timings_test::perform()", __FILE__, __LINE__,e.what());
	}

	try {
		tc3.wrong_function();
	} catch ( exception& e ) {	}
	
}

} // namespace libtensor

#ifdef SET_LIBTENSOR_TIMINGS  
	#undef LIBTENSOR_TIMINGS  
	#undef SET_LIBTENSOR_TIMINGS  
#endif