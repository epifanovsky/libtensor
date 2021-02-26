#include <cstring>
#include <libutil/exceptions/backtrace.h>
#include "backtrace_test.h"

namespace libutil {


namespace backtrace_test_ns {

bool compare_backtraces(const backtrace &bt1, const backtrace &bt2) {

    if(bt2.get_nframes() != bt1.get_nframes()) return false;
    size_t nframes = bt1.get_nframes();
    for(size_t i = 0; i < nframes; i++) {
        if(strcmp(bt1.get_frame(i), bt2.get_frame(i)) != 0) return false;
    }
    return true;
}

bool backtrace_copy_test(const backtrace &bt1) {

    // If copy is not done properly, bt2 will not be the same as bt1
    backtrace bt2 = bt1;

    return compare_backtraces(bt1, bt2);
}

} // namespace backtrace_test_ns
namespace ns = backtrace_test_ns;


void backtrace_test::perform() throw(libtest::test_exception) {

    static const char *testname = "backtrace_test::perform()";

    backtrace bt1;

    size_t nframes = bt1.get_nframes();
    for(size_t i = 0; i < nframes; i++) {
        if(strlen(bt1.get_frame(i)) == 0) {
            fail_test(testname, __FILE__, __LINE__,
                "Stack frame has a zero length.");
        }
    }

    if(!ns::backtrace_copy_test(bt1)) {
        fail_test(testname, __FILE__, __LINE__, "Backtrace copy test failed.");
    }

    backtrace bt2;
    if(ns::compare_backtraces(bt1, bt2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Expecting different backtraces.");
    }
}


} // namespace libutil
