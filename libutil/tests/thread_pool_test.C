#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>
#include <unistd.h>
#ifdef WIN32
#include <windows.h>
#endif // WIN32
#include <libutil/thread_pool/thread_pool.h>
#include <libutil/thread_pool/unknown_exception.h>
#include "thread_pool_test.h"

namespace libutil {


namespace thread_pool_test_ns { }
namespace ns = thread_pool_test_ns;


void thread_pool_test::perform() throw(libtest::test_exception) {

    srand(time(0));

    test_1a();
    test_1b();

    test_2(0, 1, 1);
    test_2(2, 1, 1);
    test_2(10, 1, 1);
    test_2(100, 1, 1);
    test_2(0, 2, 1);
    test_2(2, 2, 1);
    test_2(10, 2, 1);
    test_2(100, 2, 1);
    test_2(0, 8, 2);
    test_2(2, 8, 2);
    test_2(10, 8, 2);
    test_2(100, 8, 2);
    test_2_serial(0);
    test_2_serial(2);
    test_2_serial(10);
    test_2_serial(100);

    test_3(0, 1, 1);
    test_3(2, 1, 1);
    test_3(10, 1, 1);
    test_3(100, 1, 1);
    test_3(0, 2, 1);
    test_3(2, 2, 1);
    test_3(10, 2, 1);
    test_3(100, 2, 1);
    test_3(0, 8, 2);
    test_3(2, 8, 2);
    test_3(10, 8, 2);
    test_3(100, 8, 2);
    test_3_serial(0);
    test_3_serial(2);
    test_3_serial(10);
    test_3_serial(100);

    test_4(4, 1, 1);
    test_4(256, 1, 1);
//    test_4(1024, 1, 1);
    test_4(4, 4, 1);
    test_4(8, 4, 1);
    test_4(256, 4, 1);
    test_4(1024, 4, 1);
    test_4_serial(4);
    test_4_serial(8);
    test_4_serial(256);
    test_4_serial(1024);

    test_5(1, 1);

    test_exc_1(2, 1, 1);
    test_exc_1(10, 1, 1);
    test_exc_1(2, 2, 1);
    test_exc_1(10, 2, 1);
    test_exc_1(2, 8, 2);
    test_exc_1(10, 8, 2);
    test_exc_1_serial(2);
    test_exc_1_serial(10);
    test_exc_2(2, 1, 1);
    test_exc_2(10, 1, 1);
    test_exc_2(2, 2, 1);
    test_exc_2(10, 2, 1);
    test_exc_2(2, 8, 2);
    test_exc_2(10, 8, 2);
    test_exc_2_serial(2);
    test_exc_2_serial(10);
}


void thread_pool_test::test_1a() throw(libtest::test_exception) {

//    static const char *testname = "thread_pool_test::test_1a()";

    thread_pool tp(2, 1);
    tp.associate();
}


void thread_pool_test::test_1b() throw(libtest::test_exception) {

//    static const char *testname = "thread_pool_test::test_1b()";

    thread_pool tp(1, 1);
    tp.associate();
    tp.terminate();
}


namespace thread_pool_test_ns {

class task_2 : public task_i {
public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() { }
};

class task_iterator_2 : public task_iterator_i, public task_observer_i {
private:
    std::vector<task_2> m_tasks;
    size_t i;

public:
    task_iterator_2(size_t n) : m_tasks(n), i(0) { }
    virtual ~task_iterator_2() { }

public:
    virtual bool has_more() const {
        return i < m_tasks.size();
    }

    virtual task_i *get_next() {
        return &m_tasks[i++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
    }

};

} // namespace thread_pool_test_ns


void thread_pool_test::test_2(size_t n, size_t nthr, size_t ncpu)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_2(" << n << ", " << nthr << ", "
        << ncpu << ")";
    std::string tn = tnss.str();

    thread_pool tp(nthr, ncpu);
    tp.associate();
    ns::task_iterator_2 ti(n);
    thread_pool::submit(ti, ti);
    tp.terminate();
}


void thread_pool_test::test_2_serial(size_t n) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_2_serial(" << n << ")";
    std::string tn = tnss.str();

    ns::task_iterator_2 ti(n);
    thread_pool::submit(ti, ti);
}


namespace thread_pool_test_ns {

class task_3 : public task_i {
private:
    std::vector<int> &va, &vb, &vc;
    size_t i;

public:
    task_3(std::vector<int> &va_, std::vector<int> &vb_, std::vector<int> &vc_,
        size_t i_) : va(va_), vb(vb_), vc(vc_), i(i_) { }
    virtual ~task_3() { }

public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() {
        vc[i] = va[i] + vb[i];
    }

};

class task_iterator_3 : public task_iterator_i, public task_observer_i {
private:
    std::vector<int> &va, &vb, &vc;
    size_t i;

public:
    task_iterator_3(std::vector<int> &va_, std::vector<int> &vb_,
        std::vector<int> &vc_) : va(va_), vb(vb_), vc(vc_), i(0) { }
    virtual ~task_iterator_3() { }

public:
    virtual bool has_more() const {
        return i < va.size();
    }

    virtual task_i *get_next() {
        return new task_3(va, vb, vc, i++);
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
        delete t;
    }

};

} // namespace thread_pool_test_ns


void thread_pool_test::test_3(size_t n, size_t nthr, size_t ncpu)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_3(" << n << ", " << nthr << ", "
        << ncpu << ")";
    std::string tn = tnss.str();

    std::vector<int> va(n, 0), vb(n, 0), vc(n, 0), vc_ref(n, 0);
    for(size_t i = 0; i < n; i++) {
        va[i] = rand();
        vb[i] = rand();
        vc[i] = rand();
        vc_ref[i] = va[i] + vb[i];
    }

    thread_pool tp(nthr, ncpu);
    tp.associate();
    ns::task_iterator_3 ti(va, vb, vc);
    thread_pool::submit(ti, ti);
    tp.terminate();

    for(size_t i = 0; i < n; i++) {
        if(vc[i] != vc_ref[i]) {
            std::ostringstream ss;
            ss << "Bad result: vc[" << i << "] = " << vc[i] << ", vc_ref["
                << i << "] = " << vc_ref[i];
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
    }
}


void thread_pool_test::test_3_serial(size_t n) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_3_serial(" << n << ")";
    std::string tn = tnss.str();

    std::vector<int> va(n, 0), vb(n, 0), vc(n, 0), vc_ref(n, 0);
    for(size_t i = 0; i < n; i++) {
        va[i] = rand();
        vb[i] = rand();
        vc[i] = rand();
        vc_ref[i] = va[i] + vb[i];
    }

    ns::task_iterator_3 ti(va, vb, vc);
    thread_pool::submit(ti, ti);

    for(size_t i = 0; i < n; i++) {
        if(vc[i] != vc_ref[i]) {
            std::ostringstream ss;
            ss << "Bad result: vc[" << i << "] = " << vc[i] << ", vc_ref["
                << i << "] = " << vc_ref[i];
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
    }
}


namespace thread_pool_test_ns {

class task_4 : public task_i {
private:
    std::vector<int> &va, &vb, &vc;
    size_t ioff, n;

public:
    task_4(std::vector<int> &va_, std::vector<int> &vb_, std::vector<int> &vc_,
        size_t ioff_, size_t n_) :
        va(va_), vb(vb_), vc(vc_), ioff(ioff_), n(n_)
    { }

    virtual ~task_4() { }

public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};

class task_iterator_4 : public task_iterator_i, public task_observer_i {
private:
    std::vector<int> &va, &vb, &vc;
    size_t ioff, n1, n2;
    std::vector<task_4*> tasks;
    size_t itsk;

public:
    task_iterator_4(std::vector<int> &va_, std::vector<int> &vb_,
        std::vector<int> &vc_, size_t ioff_, size_t n1_, size_t n2_) :
        va(va_), vb(vb_), vc(vc_), ioff(ioff_), n1(n1_), n2(n2_), itsk(0) {
        tasks.push_back(new task_4(va, vb, vc, ioff, n1));
        tasks.push_back(new task_4(va, vb, vc, ioff + n1, n2));
    }
    virtual ~task_iterator_4() { }

public:
    virtual bool has_more() const {
        return itsk < tasks.size();
    }

    virtual task_i *get_next() {
        return tasks[itsk++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
        delete t;
    }

};

void task_4::perform() {

    if(n % 2) {
        for(size_t i = 0; i < n; i++) {
            vc[ioff + i] = va[ioff + i] + vb[ioff + i];
        }
    } else {
        size_t n1 = n / 2;
        task_iterator_4 ti(va, vb, vc, ioff, n1, n1);
        thread_pool::submit(ti, ti);
    }
}

} // namespace thread_pool_test_ns


void thread_pool_test::test_4(size_t n, size_t nthr, size_t ncpu)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_4(" << n << ", " << nthr << ", "
        << ncpu << ")";
    std::string tn = tnss.str();

    std::vector<int> va(n, 0), vb(n, 0), vc(n, 0), vc_ref(n, 0);
    for(size_t i = 0; i < n; i++) {
        va[i] = rand();
        vb[i] = rand();
        vc[i] = rand();
        vc_ref[i] = va[i] + vb[i];
    }

    thread_pool tp(nthr, ncpu);
    tp.associate();
    size_t n1 = n / 2, n2 = n - n1;
    ns::task_iterator_4 ti(va, vb, vc, 0, n1, n2);
    thread_pool::submit(ti, ti);
    tp.terminate();

    for(size_t i = 0; i < n; i++) {
        if(vc[i] != vc_ref[i]) {
            std::ostringstream ss;
            ss << "Bad result: vc[" << i << "] = " << vc[i] << ", vc_ref["
                << i << "] = " << vc_ref[i];
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
    }
}


void thread_pool_test::test_4_serial(size_t n) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_4_serial(" << n << ")";
    std::string tn = tnss.str();

    std::vector<int> va(n, 0), vb(n, 0), vc(n, 0), vc_ref(n, 0);
    for(size_t i = 0; i < n; i++) {
        va[i] = rand();
        vb[i] = rand();
        vc[i] = rand();
        vc_ref[i] = va[i] + vb[i];
    }

    size_t n1 = n / 2, n2 = n - n1;
    ns::task_iterator_4 ti(va, vb, vc, 0, n1, n2);
    thread_pool::submit(ti, ti);

    for(size_t i = 0; i < n; i++) {
        if(vc[i] != vc_ref[i]) {
            std::ostringstream ss;
            ss << "Bad result: vc[" << i << "] = " << vc[i] << ", vc_ref["
                << i << "] = " << vc_ref[i];
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
    }
}


namespace thread_pool_test_ns {

class task_5a : public task_i {
private:
    cond &m_sig;

public:
    task_5a(cond &sig) : m_sig(sig) { }
    virtual ~task_5a() { }

public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};

class task_5b : public task_i {
private:
    cond &m_sig;

public:
    task_5b(cond &sig) : m_sig(sig) { }
    virtual ~task_5b() { }

public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};

class task_5c : public task_i {
public:
    virtual ~task_5c() { }

public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};

class task_iterator_5a : public task_iterator_i, public task_observer_i {
private:
    cond sig;
    task_5a t5a;
    task_5b t5b;
    std::vector<task_i*> tasks;
    size_t itsk;

public:
    task_iterator_5a() : t5a(sig), t5b(sig), itsk(0) {
        tasks.push_back(&t5a);
        tasks.push_back(&t5b);
    }
    virtual ~task_iterator_5a() { }

public:
    virtual bool has_more() const {
        return itsk < tasks.size();
    }

    virtual task_i *get_next() {
        return tasks[itsk++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
    }

};

class task_iterator_5b : public task_iterator_i, public task_observer_i {
private:
    task_5c t5c;
    std::vector<task_i*> tasks;
    size_t itsk;

public:
    task_iterator_5b() : itsk(0) {
        tasks.push_back(&t5c);
    }
    virtual ~task_iterator_5b() { }

public:
    virtual bool has_more() const {
        return itsk < tasks.size();
    }

    virtual task_i *get_next() {
        return tasks[itsk++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
    }

};

void task_5a::perform() {

    task_iterator_5b ti;
    thread_pool::submit(ti, ti);
    m_sig.signal();
}

void task_5b::perform() {

    thread_pool::release_cpu();
    m_sig.wait();
    thread_pool::acquire_cpu();
}

void task_5c::perform() {

#ifdef WIN32
    Sleep(1000);
#else // WIN32
    sleep(1);
#endif // WIN32
}

} // namespace thread_pool_test_ns


void thread_pool_test::test_5(size_t nthr, size_t ncpu)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_5(" << nthr << ", " << ncpu << ")";
    std::string tn = tnss.str();

    thread_pool tp(nthr, ncpu);
    tp.associate();
    ns::task_iterator_5a ti;
    thread_pool::submit(ti, ti);
    tp.terminate();
}


namespace thread_pool_test_ns {

class task_exc_1_exception : public rethrowable_i {
public:
    virtual rethrowable_i *clone() const throw() {
        return new task_exc_1_exception;
    }
    virtual void rethrow() const {
        throw task_exc_1_exception();
    }
};

class task_exc_1_good : public task_i {
public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() {
    }
};

class task_exc_1_bad : public task_i {
public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() {
        throw task_exc_1_exception();
    }
};

class task_iterator_exc_1 : public task_iterator_i, public task_observer_i {
private:
    std::vector<task_exc_1_good> m_good_tasks;
    std::vector<task_exc_1_bad> m_bad_tasks;
    std::vector<task_i*> m_tasks;
    size_t i;

public:
    task_iterator_exc_1(size_t n) : m_good_tasks(n), m_bad_tasks(n), i(0) {
        for(size_t i = 0; i < n; i++) {
            m_tasks.push_back(&m_good_tasks[i]);
            m_tasks.push_back(&m_bad_tasks[i]);
        }
    }
    virtual ~task_iterator_exc_1() { }

public:
    virtual bool has_more() const {
        return i < m_tasks.size();
    }

    virtual task_i *get_next() {
        return m_tasks[i++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
    }

};

} // namespace thread_pool_test_ns


void thread_pool_test::test_exc_1(size_t n, size_t nthr, size_t ncpu) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_exc_1(" << n << ", " << nthr << ", "
        << ncpu << ")";
    std::string tn = tnss.str();

    thread_pool tp(nthr, ncpu);
    tp.associate();
    ns::task_iterator_exc_1 ti(n);

    bool good = false;
    try {
        thread_pool::submit(ti, ti);
    } catch(ns::task_exc_1_exception &e) {
        good = true;
    }

    tp.terminate();

    if(!good) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "Exception has not been thrown");
    }
}


void thread_pool_test::test_exc_1_serial(size_t n) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_exc_1_serial(" << n << ")";
    std::string tn = tnss.str();

    ns::task_iterator_exc_1 ti(n);

    bool good = false;
    try {
        thread_pool::submit(ti, ti);
    } catch(ns::task_exc_1_exception &e) {
        good = true;
    }

    if(!good) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "Exception has not been thrown");
    }
}


namespace thread_pool_test_ns {

class task_exc_2_good : public task_i {
public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() { }
};

class task_exc_2_bad : public task_i {
public:
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() { throw 0; }
};

class task_iterator_exc_2 : public task_iterator_i, public task_observer_i {
private:
    std::vector<task_exc_2_good> m_good_tasks;
    std::vector<task_exc_2_bad> m_bad_tasks;
    std::vector<task_i*> m_tasks;
    size_t i;

public:
    task_iterator_exc_2(size_t n) : m_good_tasks(n), m_bad_tasks(n), i(0) {
        for(size_t i = 0; i < n; i++) {
            m_tasks.push_back(&m_good_tasks[i]);
            m_tasks.push_back(&m_bad_tasks[i]);
        }
    }
    virtual ~task_iterator_exc_2() { }

public:
    virtual bool has_more() const {
        return i < m_tasks.size();
    }

    virtual task_i *get_next() {
        return m_tasks[i++];
    }

    virtual void notify_start_task(task_i *t) {
    }

    virtual void notify_finish_task(task_i *t) {
    }

};

} // namespace thread_pool_test_ns


void thread_pool_test::test_exc_2(size_t n, size_t nthr, size_t ncpu) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_exc_2(" << n << ", " << nthr << ", "
        << ncpu << ")";
    std::string tn = tnss.str();

    thread_pool tp(nthr, ncpu);
    tp.associate();
    ns::task_iterator_exc_2 ti(n);

    bool good = false;
    try {
        thread_pool::submit(ti, ti);
    } catch(unknown_exception &e) {
        // In the parallel mode exception is converted to unknown_exception
        good = true;
    }

    tp.terminate();

    if(!good) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "Exception has not been thrown");
    }
}


void thread_pool_test::test_exc_2_serial(size_t n) {

    std::ostringstream tnss;
    tnss << "thread_pool_test::test_exc_2_serial(" << n << ")";
    std::string tn = tnss.str();

    ns::task_iterator_exc_2 ti(n);

    bool good = false;
    try {
        thread_pool::submit(ti, ti);
    } catch(int) {
        // In the serial mode exception stays intact
        good = true;
    }

    if(!good) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "Exception has not been thrown");
    }
}


} // namespace libutil
