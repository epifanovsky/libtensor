#include <cmath>
#include <ctime>
#include <sstream>
#include <libutil/timings/timer.h>
#include "timer_test.h"

namespace libutil {


namespace timer_test_ns {


clock_t calc(double &d, unsigned n) {

    clock_t start = clock();
    for(unsigned i = 0; i < n; i++) d += i/10.0;
    clock_t some_time = clock();
    d -= some_time;
    return clock() - start;
}


} // namespace timer_test_ns
using namespace timer_test_ns;


void timer_test::perform() throw(libtest::test_exception) {

    static const char *testname = "timer_test::perform()";

    timer t;
    double res=0.1;

    t.start();
    clock_t duration = calc(res, 10000000);
    t.stop();
    res = (duration * 1.0)/CLOCKS_PER_SEC;
#if defined(POSIX) && !defined(HAVE_GETTIMEOFDAY)
    res -= t.duration().user_time();
    res -= t.duration().system_time();
#else
    res -= t.duration().wall_time();
#endif
    if(fabs(res) > 0.02) {
        std::ostringstream msg;
        msg << "Timer measurement not correct (diff: " << res << " s)";
        fail_test(testname, __FILE__, __LINE__, msg.str().c_str());
    }
    t.start();
    if(t.duration() != time_diff_t()) {
        fail_test(testname, __FILE__, __LINE__, "Measurement not initialized");
    }
    t.stop();
}


} // namespace libutil
