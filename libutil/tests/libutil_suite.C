#include "libutil_suite.h"

namespace libutil {


libutil_suite::libutil_suite() : libtest::test_suite("libutil") {

    add_test("backtrace", m_utf_backtrace);
    add_test("cond", m_utf_cond);
    add_test("mutex", m_utf_mutex);
    add_test("rwlock", m_utf_rwlock);
    add_test("spinlock", m_utf_spinlock);
    add_test("task_thief", m_utf_task_thief);
    add_test("thread", m_utf_thread);
    add_test("thread_pool", m_utf_thread_pool);
    add_test("timer", m_utf_timer);
    add_test("timings", m_utf_timings);
    add_test("timings_store", m_utf_timings_store);
    add_test("tls", m_utf_tls);
    add_test("version", m_utf_version);
}


} // namespace libutil
