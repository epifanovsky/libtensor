#include <libutil/thread_pool/task_i.h>
#include <libutil/thread_pool/task_thief.h>
#include "task_thief_test.h"

namespace libutil {


void task_thief_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


namespace {

class task : public task_i {
public:
    virtual ~task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform() { }
};

} // unnamed namespace


void task_thief_test::test_1() {

    static const char testname[] = "task_thief_test::test_1()";

    try {

    task_thief thief;

    std::deque<task_info> lq1, lq2, lq3, lq4;
    spinlock lqmtx1, lqmtx2, lqmtx3, lqmtx4;

    task_info tinfo;
    tinfo.tsrc = (task_source*)0xDEADBEEF;
    tinfo.tsk = (task_i*)0xDEADBEEF;

    thief.steal_task(tinfo);
    if(tinfo.tsrc != 0) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsrc != 0");
    }
    if(tinfo.tsk != 0) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != 0");
    }

    task t1, t2;
    task_i *pt1 = &t1, *pt2 = &t2;
    task_info ti1, ti2;
    ti1.tsrc = 0; ti1.tsk = &t1;
    ti2.tsrc = 0; ti2.tsk = &t2;

    thief.register_queue(lq1, lqmtx1);
    lq1.push_back(ti1);
    thief.steal_task(tinfo);
    if(!lq1.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!lq1.empty()");
    }
    if(tinfo.tsk != pt1) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != pt1");
    }

    thief.register_queue(lq2, lqmtx2);
    thief.register_queue(lq3, lqmtx3);
    thief.register_queue(lq4, lqmtx4);
    lq4.push_back(ti1);
    lq4.push_back(ti2);
    thief.steal_task(tinfo);
    //  Steals from the back
    if(tinfo.tsk != pt2) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != pt2");
    }
    thief.steal_task(tinfo);
    if(!lq4.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!lq4.empty()");
    }
    if(tinfo.tsk != pt1) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != pt1");
    }

    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void task_thief_test::test_2() {

    static const char testname[] = "task_thief_test::test_2()";

    try {

    task_thief thief;

    std::deque<task_info> lq1, lq2;
    spinlock lqmtx1, lqmtx2;

    task_info tinfo;
    tinfo.tsrc = (task_source*)0xDEADBEEF;
    tinfo.tsk = (task_i*)0xDEADBEEF;

    task t1, t2;
    task_i *pt1 = &t1, *pt2 = &t2;
    task_info ti1, ti2;
    ti1.tsrc = 0; ti1.tsk = &t1;
    ti2.tsrc = 0; ti2.tsk = &t2;

    thief.register_queue(lq1, lqmtx1);
    thief.register_queue(lq2, lqmtx2);

    lq2.push_back(ti2);
    thief.steal_task(tinfo);
    if(!lq2.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!lq2.empty()");
    }
    if(tinfo.tsk != pt2) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != pt2");
    }

    lq1.push_back(ti1);
    thief.steal_task(tinfo);
    if(!lq1.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!lq1.empty()");
    }
    if(tinfo.tsk != pt1) {
        fail_test(testname, __FILE__, __LINE__, "tinfo.tsk != pt1");
    }

    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libutil
