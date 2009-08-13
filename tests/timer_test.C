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
	res-=t.duration().m_ut*times_t::clk2sec;
	res-=t.duration().m_st*times_t::clk2sec;
	if ( fabs(res) > 0.01 ) {
		char msg[20];
		sprintf(msg, "Timer measurement not correct (diff: %6.4fs)", 
				t.duration().m_rt*times_t::clk2sec-res);
		fail_test("timer_test::perform()", __FILE__, __LINE__,msg);
	}
	t.start(); 
	if(t.duration()!=times_t()) {
		fail_test("timer_test::perform()", __FILE__, __LINE__,
			"Measurement not initialized");
	}
	t.stop();	
}

} // namespace libtensor

