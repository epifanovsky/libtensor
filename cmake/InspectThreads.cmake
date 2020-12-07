include(CheckCXXSourceCompiles)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads)
if (NOT CMAKE_USE_PTHREADS_INIT)
    message(FATAL_ERROR "We need pthreads as the threading backend.")
endif()
set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

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

#
#   Test built-in thread-local storage
#

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

# Some check to disable HAVE_CPP_DECLSPEC_THREAD in case of bugs
if(HAVE_CPP_DECLSPEC_THREAD)
    #  Intel Composer 2013 for Mac OS has a bug, use pthreads TLS
    if(APPLE AND (ICC13 OR ICC14))
    #  Intel Composer 2016 + llvm for Mac OS has bugs, use pthreads TLS
    #  (https://bugs.llvm.org/show_bug.cgi?id=25737)
        set(HAVE_CPP_DECLSPEC_THREAD OFF CACHE BOOL "" FORCE)
    elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(HAVE_CPP_DECLSPEC_THREAD OFF CACHE BOOL "" FORCE)
    endif()
endif()

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

if(NOT HAVE_CPP_DECLSPEC_THREAD AND NOT HAVE_GCC_THREAD_LOCAL)
    message(FATAL_ERROR "No way found to implement thread-local storage.")
endif()
if (HAVE_CPP_DECLSPEC_THREAD)
    add_definitions(-DHAVE_CPP_DECLSPEC_THREAD)
endif()
