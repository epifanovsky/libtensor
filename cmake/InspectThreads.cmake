include(CheckCXXSourceCompiles)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads)
if (NOT CMAKE_USE_PTHREADS_INIT)
    message(FATAL_ERROR "We need pthreads as the threading backend at the moment")
endif()
add_definitions(-DUSE_PTHREADS)

if(CMAKE_USE_PTHREADS_INIT)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

    check_cxx_source_compiles("
#include <pthread.h>
int main() {
    pthread_spinlock_t l;
    pthread_spin_init(&l, 0);
    pthread_spin_lock(&l);
    pthread_spin_unlock(&l);
    pthread_spin_destroy(&l);
    return 0;
}
" HAVE_PTHREADS_SPINLOCK)

    check_cxx_source_compiles("
#include <pthread.h>
int main() {
    pthread_mutexattr_t attr;
    pthread_mutex_t mtx;
    if(pthread_mutexattr_init(&attr) != 0) return 1;
    if(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP) != 0) return 1;
    if(pthread_mutex_init(&mtx, &attr) != 0) return 1;
    if(pthread_mutex_lock(&mtx) != 0) return 1;
    if(pthread_mutex_unlock(&mtx) != 0) return 1;
    if(pthread_mutex_destroy(&mtx) != 0) return 1;
    if(pthread_mutexattr_destroy(&attr) != 0) return 1;
    return 0;
}
" HAVE_PTHREADS_ADAPTIVE_MUTEX)

    if(HAVE_PTHREADS_SPINLOCK)
        add_definitions(-DHAVE_PTHREADS_SPINLOCK)
    endif(HAVE_PTHREADS_SPINLOCK)

    if(HAVE_PTHREADS_ADAPTIVE_MUTEX)
        add_definitions(-DHAVE_PTHREADS_ADAPTIVE_MUTEX)
    endif(HAVE_PTHREADS_ADAPTIVE_MUTEX)

    if(NOT HAVE_PTHREADS_SPINLOCK AND APPLE)

        check_cxx_source_compiles("
#include <libkern/OSAtomic.h>
int main() {
    OSSpinLock l = OS_SPINLOCK_INIT;
    OSSpinLockLock(&l);
    OSSpinLockUnlock(&l);
    return 0;
}
" HAVE_MACOS_SPINLOCK)

        if(HAVE_MACOS_SPINLOCK)
            add_definitions(-DHAVE_MACOS_SPINLOCK)
        endif(HAVE_MACOS_SPINLOCK)
    endif(NOT HAVE_PTHREADS_SPINLOCK AND APPLE)

endif()

#   Test built-in thread-local storage

#    Test Intel-style TLS
check_cxx_source_compiles("
#if defined(__CYGWIN__)
#error Cygwin g++ does not support __declspec(thread)
#endif
int main() {
    __declspec(thread) static int a;
    return 0;
}
" HAVE_CPP_DECLSPEC_THREAD)

#    Test GCC-style TLS
#    Intel 11.0 compiler has a bug that doesn't allow static
#    thread-local members in templates
check_cxx_source_compiles("
int main() {
    static __thread int a;
    return 0;
}
template<typename T> class C { static __thread int a; };
template<typename T> __thread int C<T>::a;
" HAVE_GCC_THREAD_LOCAL)

if(HAVE_CPP_DECLSPEC_THREAD)
    #  Intel Composer 2013 for Mac OS has a bug, use pthreads TLS
    if(APPLE AND (ICC13 OR ICC14))
    #  Intel Composer 2016 + llvm for Mac OS has bugs, use pthreads TLS
    #  (https://bugs.llvm.org/show_bug.cgi?id=25737)
    elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    elseif(CLANG)
        if(HAVE_GCC_THREAD_LOCAL)
            add_definitions(-DHAVE_GCC_THREAD_LOCAL -DUSE_BUILTIN_TLS)
        endif()
    else()
        add_definitions(-DHAVE_CPP_DECLSPEC_THREAD -DUSE_BUILTIN_TLS)
    endif()
elseif(HAVE_GCC_THREAD_LOCAL)
    add_definitions(-DUSE_BUILTIN_TLS -DHAVE_GCC_THREAD_LOCAL)
endif()
