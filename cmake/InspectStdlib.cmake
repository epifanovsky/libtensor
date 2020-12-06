#
#   Random-number generator
#
check_cxx_source_compiles("
#include<cstdlib>
int main() {
  double d = drand48();
  return 0;
}
" HAVE_DRAND48)

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
