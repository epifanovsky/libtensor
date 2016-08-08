set(SRC
    version.C
    exceptions/backtrace.C
    exceptions/exception.C
    exceptions/rethrowable_i.C
    thread_pool/task_source.C
    thread_pool/task_source_mpi.C
    thread_pool/task_thief.C
    thread_pool/thread_pool.C
    thread_pool/unknown_exception.C
    thread_pool/worker.C
    timings/local_timings_store_base.C
    timings/timings_store.C
    timings/timer.C
)

set(SRC_THREADS_WINDOWS
    threads/windows/tls_windows.C
)

set(SRC_THREADS_POSIX
    threads/posix/cond_posix.C
    threads/posix/mutex_posix.C
    threads/posix/rwlock_posix.C
    threads/posix/thread_posix.C
    threads/posix/tls_posix.C
)

if(CMAKE_USE_WIN32_THREADS_INIT)
    set(SRC_THREADS ${SRC_THREADS_WINDOWS})
elseif(CMAKE_USE_PTHREADS_INIT)
    set(SRC_THREADS ${SRC_THREADS_POSIX})
    if(HAVE_PTHREADS_SPINLOCK)
        set(SRC_THREADS ${SRC_THREADS} threads/posix/spinlock_posix.C)
    elseif(HAVE_MACOS_SPINLOCK)
        set(SRC_THREADS ${SRC_THREADS} threads/macos/spinlock_macos.C)
    endif()
endif()

add_library(util ${SRC} ${SRC_THREADS})

include_directories(${LIBUTIL_DIR})

set_target_properties(util PROPERTIES
    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:LIBUTIL_DEBUG>)
