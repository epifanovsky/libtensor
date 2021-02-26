#include <iostream>
#include <sstream>
#include <libutil/version.h>
#include "libutil_suite.h"

using namespace std;
using namespace libutil;
using libtest::test_exception;


class suite_handler : public libtest::suite_event_handler {
public:
    virtual void on_suite_start(const char *suite) {

    }

    virtual void on_suite_end(const char *suite) {

    }

    virtual void on_test_start(const char *test) {

        std::cout << "Test " << test << " ... ";
    }

    virtual void on_test_end_success(const char *test) {

        std::cout << "done." << std::endl;
    }

    virtual void on_test_end_exception(const char *test,
        const test_exception &e) {

        std::cout << "FAIL!" << std::endl << e.what() << std::endl;
    }

};


int main(int argc, char **argv) {

    ostringstream ss;
    ss << " Unit tests for libutil-" << version::get_string() << " ";
    string separator(ss.str().size(), '-');
    cout << separator << endl << ss.str() << endl << separator << endl;

    suite_handler handler;
    libutil_suite suite;
    suite.set_handler(&handler);

    if(argc == 1) {
        suite.run_all_tests();
    } else {
        for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
    }
}
