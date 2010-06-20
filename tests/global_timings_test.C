#include <libtensor/timer.h>
#include <libtensor/global_timings.h>
#include "global_timings_test.h"

namespace libtensor {

void global_timings_test::wait( double seconds )  {
	clock_t end=clock()+ CLOCKS_PER_SEC*seconds;
	while ( clock() < end ) {}
}


void global_timings_test::perform() throw(libtest::test_exception) {
	timer t1, t2, t3;
	time_diff_t r1, r2, r3;
	global_timings& gt=global_timings::get_instance();

	t1.start(); wait( 0.1 ); t1.stop();
	r1=t1.duration(); gt.add_to_timer("t1",t1);

	t2.start(); wait(0.2); t2.stop();
	r2=t2.duration(); gt.add_to_timer("t2",t2);

	r1+=t2.duration(); gt.add_to_timer("t1",t2);

	t3.start(); wait(0.3); t3.stop();
	r3=t3.duration(); gt.add_to_timer("t3",t3);

	try {
		if ( gt.get_time("t1") != r1 )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Wrong time of t1");
		if ( gt.get_time("t2") != r2 )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Wrong time of t2");
		if ( gt.get_time("t3") != r3 )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Wrong time of t3");
	} catch ( exception& e ) {
		fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Unknown timer name");
	}

	gt.reset();
	try {
		if ( gt.get_time("t1") != time_diff_t() )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Timer t1 not deleted");
		if ( gt.get_time("t2") != time_diff_t() )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Timer t2 not deleted");
		if ( gt.get_time("t3") != time_diff_t() )
			fail_test("global_timings_test::perform()", __FILE__, __LINE__,
				"Timer t3 not deleted");
	} catch ( exception& e ) {
	}
}

} // namespace libtensor

