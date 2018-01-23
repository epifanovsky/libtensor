#include <cstdlib>
#include <libutil/threads/thread.h>
#include "thread_test.h"

namespace libutil {


void thread_test::perform() throw(libtest::test_exception) {

    test_1();
}


namespace thread_test_ns {

class thr_test_1 : public thread {
private:
    bool m_run;
public:
    thr_test_1() : m_run(false) { }
    virtual ~thr_test_1() { }
    bool get_run() const {
        return m_run;
    }
public:
    virtual void run() {
        m_run = true;
    }
};

} // namespace thread_test_ns
namespace ns = thread_test_ns;


void thread_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "thread_test::test_1()";

    ns::thr_test_1 thr;
    thr.start();
    thr.join();

    if(!thr.get_run()) {
        fail_test(testname, __FILE__, __LINE__, "The thread did not run.");
    }
}


} // namespace libutil
