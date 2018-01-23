#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <libutil/threads/mutex.h>
#include <libutil/threads/thread.h>
#include "mutex_test.h"

namespace libutil {


namespace mutex_test_ns {

struct test_struct {
    volatile int a, b, c;
    test_struct() : a(0), b(0), c(0) { }
};

class test_thr : public thread {
private:
    test_struct &m_s;
    mutex &m_lock;
    unsigned m_wait;
    volatile bool m_started;
    volatile bool m_done;
    bool m_ok;

public:
    test_thr(test_struct &s, mutex &lock, unsigned wait = 0) :
        m_s(s), m_lock(lock), m_wait(wait), m_started(false),
        m_done(false), m_ok(true) { }
    virtual ~test_thr() { }
    bool is_started() const { return m_started; }
    void finish() { m_done = true; }
    bool is_done() const { return m_done; }
    bool is_ok() const { return m_ok; }

public:
    virtual void run() {
        m_started = true;
        while(!m_done) {
            if(m_wait > 0) usleep(m_wait);
            m_lock.lock();
            int a = m_s.a, b = m_s.b, c = m_s.c;
            if(a != b || a != c) m_ok = false;
            m_lock.unlock();
        }
    }
};

} // namespace mutex_test_ns
namespace ns = mutex_test_ns;


void mutex_test::perform() throw(libtest::test_exception) {

    test_1a();
    test_1b();
    test_2();
}


/** \test Checks that race condition is observed without mutexes
        (mutex is disabled while modifying the variable)
 **/
void mutex_test::test_1a() throw(libtest::test_exception) {

    static const char *testname = "mutex_test::test_1a()";

//    try {

    ns::test_struct s;
    mutex lock;
    ns::test_thr thr1(s, lock), thr2(s, lock);
    thr1.start(); thr2.start();
    usleep(500);
    while(!thr1.is_started() || !thr2.is_started());
    for(size_t i = 0; i < 200; i++) {
        usleep(20);
        s.a++;
        usleep(10);
        s.b++;
        usleep(30);
        s.c++;
    }
    thr1.finish(); thr2.finish();
    thr1.join(); thr2.join();

    bool ok1 = thr1.is_ok(), ok2 = thr2.is_ok();
    if(ok1 || ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed to generate race condition.");
    }

//    } catch(vmm_exception &e) {
//        fail_test(testname, __FILE__, __LINE__, e.what());
//    }
}


/** \test Checks that race condition is observed without mutexes
        (mutex is disabled while verifying the variable)
 **/
void mutex_test::test_1b() throw(libtest::test_exception) {

    static const char *testname = "mutex_test::test_1b()";

//    try {

    ns::test_struct s;
    mutex lock, lock1, lock2;
    ns::test_thr thr1(s, lock1), thr2(s, lock2);
    thr1.start(); thr2.start();
    usleep(500);
    while(!thr1.is_started() || !thr2.is_started());
    for(size_t i = 0; i < 200; i++) {
        usleep(20);
        lock.lock();
        s.a++;
        usleep(10);
        s.b++;
        usleep(30);
        s.c++;
        lock.unlock();
    }
    thr1.finish(); thr2.finish();
    thr1.join(); thr2.join();

    bool ok1 = thr1.is_ok(), ok2 = thr2.is_ok();
    if(ok1 || ok2) {
        fail_test(testname, __FILE__, __LINE__,
            "Failed to generate race condition.");
    }

//    } catch(vmm_exception &e) {
//        fail_test(testname, __FILE__, __LINE__, e.what());
//    }
}


/** \test One thread increments a counter using a mutex, two other
        threads verify
 **/
void mutex_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "mutex_test::test_2()";

//    try {

    ns::test_struct s;
    mutex lock;
    ns::test_thr thr1(s, lock), thr2(s, lock);
    thr1.start(); thr2.start();
    usleep(500);
    while(!thr1.is_started() || !thr2.is_started());
    for(size_t i = 0; i < 200; i++) {
        usleep(20);
        lock.lock();
        s.a++;
        usleep(10);
        s.b++;
        usleep(30);
        s.c++;
        lock.unlock();
    }
    thr1.finish(); thr2.finish();
    thr1.join(); thr2.join();

    bool ok1 = thr1.is_ok(), ok2 = thr2.is_ok();
    if(!ok1 || !ok2) {
        fail_test(testname, __FILE__, __LINE__, "Mutex failed.");
    }

//    } catch(vmm_exception &e) {
//        fail_test(testname, __FILE__, __LINE__, e.what());
//    }
}


} // namespace libutil
