set(SRC
    version.C
    exceptions/backtrace.C
    exceptions/exception.C
    exceptions/rethrowable_i.C
    thread_pool/task_source.C
    thread_pool/task_thief.C
    thread_pool/thread_pool.C
    thread_pool/unknown_exception.C
    thread_pool/worker.C
    timings/local_timings_store_base.C
    timings/timings_store.C
    timings/timer.C
)
if(HAVE_EXECINFO_BACKTRACE)
    set_property(SOURCE exceptions/backtrace.C
                 APPEND PROPERTY COMPILE_DEFINITIONS HAVE_EXECINFO_BACKTRACE=1)
endif()

set(SRC_THREADS
    threads/posix/cond_posix.C
    threads/posix/mutex_posix.C
    threads/posix/rwlock_posix.C
    threads/posix/thread_posix.C
    threads/posix/tls_posix.C
)

if(APPLE)
    set(SRC_THREADS ${SRC_THREADS} threads/macos/spinlock_macos.C)
else()
    set(SRC_THREADS ${SRC_THREADS} threads/posix/spinlock_posix.C)
endif()

add_library(util OBJECT ${SRC} ${SRC_THREADS})
target_include_directories(util PUBLIC ${PROJECT_SOURCE_DIR})

if (HAVE_PTHREADS_ADAPTIVE_MUTEX)
    target_compile_definitions(util PRIVATE HAVE_PTHREADS_ADAPTIVE_MUTEX=1)
endif()

set_target_properties(util PROPERTIES
    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBUTIL_DEBUG>)
