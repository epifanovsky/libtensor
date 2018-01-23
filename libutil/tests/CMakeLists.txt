set(SRC
    libutil_tests.C
    libutil_suite.C
    backtrace_test.C
    cond_test.C
    mutex_test.C
    rwlock_test.C
    spinlock_test.C
    task_thief_test.C
    thread_test.C
    thread_pool_test.C
    timer_test.C
    timings_test.C
    timings_store_test.C
    tls_test.C
    version_test.C
)
add_executable(libutil_tests ${SRC})

#include(${LIBTEST_DIR}/cmake/SetupLibTest.txt)

include_directories(${LIBUTIL_DIR} ${LIBTEST_DIR})

set_target_properties(libutil_tests PROPERTIES
    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBUTIL_DEBUG>)

target_link_libraries(libutil_tests util ${LIBUTIL_DEP_LIBS} libtest)
