#ifndef LIBUTIL_LIBUTIL_SUITE_H
#define LIBUTIL_LIBUTIL_SUITE_H

#include <libtest/test_suite.h>
#include "backtrace_test.h"
#include "cond_test.h"
#include "mutex_test.h"
#include "rwlock_test.h"
#include "spinlock_test.h"
#include "task_thief_test.h"
#include "thread_test.h"
#include "thread_pool_test.h"
#include "timer_test.h"
#include "timings_test.h"
#include "timings_store_test.h"
#include "tls_test.h"
#include "version_test.h"

using libtest::unit_test_factory;

/** \defgroup libutil_tests Tests
    \ingroup libutil
 **/

namespace libutil {


/** \brief Test suite for the utility library (libutil)

    This suite runs the following tests:
     - libutil::backtrace_test
     - libutil::cond_test
     - libutil::mutex_test
     - libutil::rwlock_test
     - libutil::spinlock_test
     - libutil::task_thief_test
     - libutil::thread_test
     - libutil::thread_pool_test
     - libutil::timer_test
     - libutil::timings_test
     - libutil::timings_store_test
     - libutil::tls_test
     - libutil::version_test

    \ingroup libutil_tests
 **/
class libutil_suite: public libtest::test_suite {
private:
    unit_test_factory<backtrace_test> m_utf_backtrace;
    unit_test_factory<cond_test> m_utf_cond;
    unit_test_factory<mutex_test> m_utf_mutex;
    unit_test_factory<rwlock_test> m_utf_rwlock;
    unit_test_factory<spinlock_test> m_utf_spinlock;
    unit_test_factory<task_thief_test> m_utf_task_thief;
    unit_test_factory<thread_test> m_utf_thread;
    unit_test_factory<thread_pool_test> m_utf_thread_pool;
    unit_test_factory<timer_test> m_utf_timer;
    unit_test_factory<timings_test> m_utf_timings;
    unit_test_factory<timings_store_test> m_utf_timings_store;
    unit_test_factory<tls_test> m_utf_tls;
    unit_test_factory<version_test> m_utf_version;

public:
    //! Creates the suite
    libutil_suite();

};


} // namespace libutil

#endif // LIBUTIL_LIBUTIL_SUITE_H
