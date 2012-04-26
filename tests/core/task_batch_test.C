#include <sstream>
#include <libtensor/mp/task_batch.h>
#include <libtensor/mp/worker_pool.h>
#include "task_batch_test.h"

namespace libtensor {


void task_batch_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_exc_1();
}


namespace task_batch_test_ns {

class task_1 : public task_i {
private:
    int m_val;
    int m_newval;
    unsigned m_wait;

public:
    task_1(int oldval, int newval, unsigned wait = 100) :
        m_val(oldval), m_newval(newval), m_wait(wait) { }
    virtual ~task_1() { }
    virtual void perform(cpu_pool&) throw(exception) {
        usleep(m_wait);
        m_val = m_newval;
    }
    int get_val() {
        return m_val;
    }
};

class task_2 : public task_i {
private:
    size_t m_n;
    bool m_ok;

public:
    task_2(size_t n) : m_n(n), m_ok(false) { }
    virtual ~task_2() { }
    virtual void perform(cpu_pool&) throw(exception) {

        std::vector<task_1*> t(m_n);
        for(size_t i = 0; i < m_n; i++) {
            t[i] = new task_1(i * 2, i * 2 + 1, 10);
        }
        task_batch b;
        for(size_t i = 0; i < m_n; i++) {
            b.push(*t[i]);
        }
        b.wait();
        m_ok = true;
        for(size_t i = 0; i < m_n; i++) {
            if(t[i]->get_val() != i * 2 + 1) m_ok = false;
        }
        for(size_t i = 0; i < m_n; i++) {
            delete t[i];
        }
    }
    bool get_ok() {
        return m_ok;
    }
};

class task_3 : public task_i {
public:
    virtual ~task_3() { }
    virtual void perform(cpu_pool&) throw(exception) {
        throw bad_parameter("task_batch_test_ns", "task_3", "perform()",
            __FILE__, __LINE__, "a");
    }
};

} // namespace task_batch_test_ns


void task_batch_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "task_batch_test::test_1()";

    try {

    task_batch_test_ns::task_1 t1(10, 20), t2(100, 200), t3(1000, 2000);
    task_batch b;
    b.push(t1);
    b.push(t2);
    b.push(t3);
    b.wait();
    if(t3.get_val() != 2000) {
        fail_test(testname, __FILE__, __LINE__, "Task 3 failed.");
    }
    if(t2.get_val() != 200) {
        fail_test(testname, __FILE__, __LINE__, "Task 2 failed.");
    }
    if(t1.get_val() != 20) {
        fail_test(testname, __FILE__, __LINE__, "Task 1 failed.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void task_batch_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "task_batch_test::test_2()";

    try {

    size_t ntasks = 1000;
    std::vector<task_batch_test_ns::task_1*> t(ntasks);
    for(size_t i = 0; i < ntasks; i++) {
        t[i] = new task_batch_test_ns::task_1(2 * i, 2 * i + 1);
    }

    task_batch b;
    for(size_t i = 0; i < ntasks; i++) b.push(*t[i]);
    b.wait();

    for(size_t i = 0; i < ntasks; i++) {
        if(t[i]->get_val() != 2 * i + 1) {
            std::ostringstream ss;
            ss << "Task " << i + 1 << " failed.";
            fail_test(testname, __FILE__, __LINE__,
                ss.str().c_str());
        }
    }

    for(size_t i = 0; i < ntasks; i++) {
        delete t[i];
    }
    t.clear();

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void task_batch_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "task_batch_test::test_3()";

    try {

    task_batch_test_ns::task_2 t1(500), t2(100), t3(20), t4(1100);
    task_batch b;
    b.push(t1);
    b.push(t2);
    b.push(t3);
    b.push(t4);
    b.wait();
    if(!t1.get_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Task 1 failed.");
    }
    if(!t2.get_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Task 2 failed.");
    }
    if(!t3.get_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Task 3 failed.");
    }
    if(!t4.get_ok()) {
        fail_test(testname, __FILE__, __LINE__, "Task 3 failed.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void task_batch_test::test_exc_1() throw(libtest::test_exception) {

    static const char *testname = "task_batch_test::test_exc_1()";

    try {

    task_batch_test_ns::task_1 t1(10, 20), t2(100, 200), t3(1000, 2000);
    task_batch_test_ns::task_3 t1err;
    task_batch b;
    b.push(t1);
    b.push(t2);
    b.push(t3);
    b.push(t1err);

    try {
        b.wait();
    } catch(bad_parameter &e) {
    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, "Bad exception type.");
    } catch(...) {
        fail_test(testname, __FILE__, __LINE__,
            "Unknown exception type.");
    }

    if(t3.get_val() != 2000) {
        fail_test(testname, __FILE__, __LINE__, "Task 3 failed.");
    }
    if(t2.get_val() != 200) {
        fail_test(testname, __FILE__, __LINE__, "Task 2 failed.");
    }
    if(t1.get_val() != 20) {
        fail_test(testname, __FILE__, __LINE__, "Task 1 failed.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
