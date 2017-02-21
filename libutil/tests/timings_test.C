#include <libutil/timings/timings.h>
#include <libutil/timings/timings_store.h>
#include "timings_test.h"

namespace libutil {

namespace timings_test_ns {


struct timed_module { };

class timed_class : public timings<timed_class, timed_module, true>  {
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

class timed_class2 : public timings<timed_class2, timed_module, true>  {
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
using namespace timings_test_ns;


void timings_test::perform() throw(libtest::test_exception) {

    timings_store<timed_module>::get_instance().reset();

	timed_class tc1, tc2;

//	try {
		tc1.some_function();
		tc1.some_function();
		tc2.some_function();
		timings_store<timed_module>::get_instance().get_time("timed_class");


//	} catch ( exception& e ) {
//		fail_test("timings_test::perform()",__FILE__,__LINE__,e.what());
//	}

	timed_class2 tc3;

//	try {
		tc3.some_function();
		tc3.some_other_function();
		tc3.some_function();

		timings_store<timed_module>::get_instance().get_time(
		    "timed_class2::some_function()");
		timings_store<timed_module>::get_instance().get_time(
		    "timed_class2::some_other_function()");
//	} catch ( exception& e ) {
//		fail_test("timings_test::perform()", __FILE__, __LINE__,e.what());
//	}

//	try {
//		tc3.wrong_function();
//		fail_test("timings_test::perform()", __FILE__, __LINE__,"No exception thrown!");
//	} catch ( exception& e ) {	}

}


} // namespace libutil
