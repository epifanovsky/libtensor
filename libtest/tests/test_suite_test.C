#include "test_suite_test.h"

namespace libtest {

bool test_suite_test::test_1::m_ran = false;
bool test_suite_test::test_2::m_ran = false;


void test_suite_test::perform() throw(test_exception) {

    static const char testname[] = "test_suite_test::perform()";

    test_suite_impl suite;

    if(suite.get_num_tests() != 2) {
        fail_test(testname, __FILE__, __LINE__, "get_num_tests() failed.");
    }

    suite.run_test("test_1");
    if(!(test_1::ran() && !test_2::ran())) {
        fail_test(testname, __FILE__, __LINE__, "run_test(\"test_1\") failed.");
    }
    test_1::reset();
    suite.run_test("test_2");
    if(!(!test_1::ran() && test_2::ran())) {
        fail_test(testname, __FILE__, __LINE__, "run_test(\"test_2\") failed.");
    }
    test_2::reset();
    suite.run_test("test_1");
    suite.run_test("test_2");
    if(!(test_1::ran() && test_2::ran())) {
        fail_test(testname, __FILE__, __LINE__,
            "run_test(\"test_1\"); run_test(\"test_2\") failed.");
    }
    test_1::reset();
    test_2::reset();
    suite.run_all_tests();
    if(!(test_1::ran() && test_2::ran())) {
        fail_test(testname, __FILE__, __LINE__, "run_all_tests() failed.");
    }
}


} // namespace libtest

