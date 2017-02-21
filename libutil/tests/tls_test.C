#include <cstdlib>
#include <unistd.h>
#include <libutil/threads/tls.h>
#include <libutil/threads/thread.h>
#include "tls_test.h"


#ifdef __MINGW32__
#include <windows.h>
#define usleep(usec) (Sleep((usec) / 1000), 0)
#endif // __MINGW32__


namespace libutil {


namespace tls_test_ns {

class thread_1 : public thread {
private:
    int m_n1, m_n2;
    bool m_ok;
public:
    thread_1(int n1, int n2) : m_n1(n1), m_n2(n2), m_ok(false) { }
    virtual ~thread_1() { }
public:
    virtual void run() {
        tls<int>::get_instance().get() = m_n1;
        usleep(100);
        if(tls<int>::get_instance().get() != m_n1) {
            return;
        }
        tls<int>::get_instance().get() = m_n2;
        usleep(20);
        if(tls<int>::get_instance().get() != m_n2) {
            return;
        }
        m_ok = true;
    }
    bool is_ok() const {
        return m_ok;
    }
};

} // namespace tls_test_ns


void tls_test::perform() throw(libtest::test_exception) {

    test_1();
}


void tls_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "tls_test::test_1()";

    tls<int>::get_instance().get() = 115;

    tls_test_ns::thread_1 t1(1, 2), t2(5, 6), t3(-10, -20);
    t1.start();
    t2.start();
    t3.start();
    t1.join();
    t2.join();
    t3.join();

    if(tls<int>::get_instance().get() != 115) {
        fail_test(testname, __FILE__, __LINE__, "Main thread failed.");
    }
    if(!t1.is_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Thread 1 failed.");
    }
    if(!t2.is_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Thread 2 failed.");
    }
    if(!t3.is_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Thread 3 failed.");
    }
}


} // namespace libutil
