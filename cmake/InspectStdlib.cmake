#
# Random-number generator
#
check_cxx_source_compiles("
#include<cstdlib>
int main() {
  double d = drand48();
  return 0;
}
" HAVE_DRAND48)
if(HAVE_DRAND48)
    add_definitions(-DHAVE_DRAND48)
endif(HAVE_DRAND48)

#
#   Timer functions
#
check_cxx_source_compiles("
#include <sys/time.h>
int main() {
    struct timeval t;
    gettimeofday(&t, 0);
    return 0;
}
" HAVE_GETTIMEOFDAY)

if(HAVE_GETTIMEOFDAY)
    add_definitions(-DHAVE_GETTIMEOFDAY)
endif(HAVE_GETTIMEOFDAY)

check_cxx_source_compiles("
#include <sys/times.h>
int main() {
    struct tms t;
    clock_t c = times(&t);
    return 0;
}
" HAVE_SYS_TIMES)

if(HAVE_SYS_TIMES)
    add_definitions(-DHAVE_SYS_TIMES)
endif(HAVE_SYS_TIMES)

#
#   Test stack tracing capability
#

check_cxx_source_compiles("
#include <cstdlib>
#include <execinfo.h>
int main() {
    void *array[100];
    int size;
    char **symbols;
    size = backtrace(array, 100);
    symbols = backtrace_symbols(array, size);
    free(symbols);
}
" HAVE_EXECINFO_BACKTRACE)

if(HAVE_EXECINFO_BACKTRACE)
    add_definitions(-DHAVE_EXECINFO_BACKTRACE)
endif(HAVE_EXECINFO_BACKTRACE)
