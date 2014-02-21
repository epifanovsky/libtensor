#include <iostream>
#include <sstream>
#include <libtensor/version.h>
#include <libtensor/linalg/linalg.h>
#include "libtensor_linalg_suite.h"


using namespace libtensor;
using namespace std;
using libtest::test_exception;


class suite_handler : public libtest::suite_event_handler {
public:
    virtual void on_suite_start(const char *suite) {

    }

    virtual void on_suite_end(const char *suite) {

    }

    virtual void on_test_start(const char *test) {
        cout << "Test " << test << " ... ";
    }

    virtual void on_test_end_success(const char *test) {
        cout << "done." << endl;
    }

    virtual void on_test_end_exception(const char *test,
        const test_exception &e) {

        cout << "FAIL!" << endl << e.what() << endl;
    }

};


int main(int argc, char **argv) {

    linalg::rng_setup(0);

    ostringstream ss1, ss2;
    ss1 << " Unit tests for libtensor " << version::get_string() << ". ";
    ss2 << " Linear algebra test suite. ";
    string separator(std::max(ss1.str().size(), ss2.str().size()), '-');
    cout << separator << endl << ss1.str() << endl << ss2.str() << endl
        << separator << endl;

    suite_handler handler;
    libtensor_linalg_suite suite;
    suite.set_handler(&handler);

    if(argc == 1) {
        suite.run_all_tests();
    } else {
        for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
    }
}
