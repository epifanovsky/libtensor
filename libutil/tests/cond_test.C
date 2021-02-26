#include <cstdlib>
#include <unistd.h>
#include <libutil/exceptions/util_exceptions.h>
#include <libutil/threads/cond.h>
#include <libutil/threads/thread.h>
#include "cond_test.h"

namespace libutil {


namespace cond_test_ns {

class test_thr : public thread {
private:
    cond &m_c;
    unsigned m_wait;
    volatile bool m_started;
    volatile bool m_done;

public:
    test_thr(cond &c, unsigned wait = 0) :
        m_c(c), m_wait(wait), m_started(false), m_done(false) { }
    virtual ~test_thr() { }
    bool is_started() const { return m_started; }
    bool is_done() const { return m_done; }

public:
    virtual void run() {
        m_started = true;
        if(m_wait > 0) usleep(m_wait);
        m_done = true;
        m_c.signal();
    }
};

} // namespace cond_test_ns
namespace ns = cond_test_ns;


void cond_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
}


/** \test A condition is created, but never used.
 **/
void cond_test::test_1() throw(libtest::test_exception) {

//    static const char *testname = "cond_test::test_1()";

//    try {

    cond c;

//    } catch(vmm_exception &e) {
//        fail_test(testname, __FILE__, __LINE__, e.what());
//    }
}


/**	\test First wait(), then signal().
 **/
void cond_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_2()";

    bool ok;

    try {

    cond c;
    ns::test_thr thr(c, 1000);
    thr.start();
    while(!thr.is_started());
    c.wait();
    ok = thr.is_done();
    thr.join();

    } catch(threads_exception &e) {
    	fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Condition variable malfunction.");
    }
}


/** \test First signal(), then wait().
 **/
void cond_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_3()";

    bool ok;

    try {

    cond c;
    ns::test_thr thr(c, 0);
    thr.start();
    while(!thr.is_started());
    usleep(1000);
    c.wait();
    ok = thr.is_done();
    thr.join();

    } catch(threads_exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Condition variable malfunction.");
    }
}


/** \test First wait(), then signal() twice.
 **/
void cond_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_4()";

    bool ok1, ok2;

    try {

    cond c;
    ns::test_thr thr1(c, 1000);
    thr1.start();
    while(!thr1.is_started());
    c.wait();
    ok1 = thr1.is_done();
    thr1.join();

    ns::test_thr thr2(c, 1000);
    thr2.start();
    while(!thr2.is_started());
    c.wait();
    ok2 = thr2.is_done();
    thr2.join();

    } catch(threads_exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok1) {
        fail_test(testname, __FILE__, __LINE__,
            "Single-use condition failure.");
    }
    if(ok1 && !ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Double-use condition failure.");
    }
}


/** \test First signal(), then wait() twice.
 **/
void cond_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_5()";

    bool ok1, ok2;

    try {

    cond c;
    ns::test_thr thr1(c, 0);
    thr1.start();
    while(!thr1.is_started());
    usleep(1000);
    c.wait();
    ok1 = thr1.is_done();
    thr1.join();

    ns::test_thr thr2(c, 0);
    thr2.start();
    while(!thr2.is_started());
    usleep(1000);
    c.wait();
    ok2 = thr2.is_done();
    thr2.join();

    } catch(threads_exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok1) {
        fail_test(testname, __FILE__, __LINE__,
            "Single-use condition failure.");
    }
    if(ok1 && !ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Double-use condition failure.");
    }
}


/** \test First wait()-signal() cycle, then signal()-wait().
 **/
void cond_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_6()";

    bool ok1, ok2;

    try {

    cond c;
    ns::test_thr thr1(c, 1000);
    thr1.start();
    while(!thr1.is_started());
    c.wait();
    ok1 = thr1.is_done();
    thr1.join();

    ns::test_thr thr2(c, 0);
    thr2.start();
    while(!thr2.is_started());
    usleep(1000);
    c.wait();
    ok2 = thr2.is_done();
    thr2.join();

    } catch(threads_exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok1) {
        fail_test(testname, __FILE__, __LINE__,
            "Single-use condition failure.");
    }
    if(ok1 && !ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Double-use condition failure.");
    }
}


/** \test First signal()-wait() cycle, then wait()-signal().
 **/
void cond_test::test_7() throw(libtest::test_exception) {

    static const char *testname = "cond_test::test_7()";

    bool ok1, ok2;

    try {

    cond c;
    ns::test_thr thr1(c, 0);
    thr1.start();
    while(!thr1.is_started());
    usleep(1000);
    c.wait();
    ok1 = thr1.is_done();
    thr1.join();

    ns::test_thr thr2(c, 1000);
    thr2.start();
    while(!thr2.is_started());
    c.wait();
    ok2 = thr2.is_done();
    thr2.join();

    } catch(threads_exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if(!ok1) {
        fail_test(testname, __FILE__, __LINE__,
            "Single-use condition failure.");
    }
    if(ok1 && !ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Double-use condition failure.");
    }
}


} // namespace libutil
