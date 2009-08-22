#include <sstream>
#include <libtensor.h>
#include "timer_test.h"
#include "../timer.h"

namespace libtensor {
clock_t timer_test::calc( double& d, unsigned int n )  {
	clock_t start=clock();
	for (unsigned int i=0; i<n; i++) {
		d+=i/10.0;
	}
	clock_t some_time=clock();
	d-=some_time;
	
	return clock()-start; 
}

void timer_test::perform() throw(libtest::test_exception) {
	timer t;
	double res=0.1;

	t.start();
	clock_t duration=calc(res,10000000);
	t.stop(); 
	res=(duration*1.0)/CLOCKS_PER_SEC;
//#ifdef POSIX
//	res-=t.duration().user_time();
//	res-=t.duration().system_time();
//#else
	res-=t.duration().wall_time();
//#endif
	if ( fabs(res) > 0.01 ) {
		std::ostringstream msg;
		msg << "Timer measurement not correct (diff: " << res << "fs)";
		fail_test("timer_test::perform()", __FILE__, __LINE__,msg.str().c_str());
	}
	t.start(); 
	if(t.duration()!=time_diff_t()) {
		fail_test("timer_test::perform()", __FILE__, __LINE__,
			"Measurement not initialized");
	}
	t.stop();	
}

} // namespace libtensor

