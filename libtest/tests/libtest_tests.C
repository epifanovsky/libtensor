#include <iostream>
#include <string>
#include "libtest_suite.h"


using namespace libtest;


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

    std::string smsg("Performing tests for libtest");
    std::string sline(smsg.size(), '-');
    std::cout << sline << std::endl << smsg << std::endl << sline << std::endl;

    suite_handler handler;
    libtest_suite suite;
    suite.set_handler(&handler);

    if(argc == 1) {
        suite.run_all_tests();
    } else {
        for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
    }
}

