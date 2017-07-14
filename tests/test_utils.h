#include <iostream>

int fail_test(
    const std::string &testname, const std::string &source, unsigned lineno,
    const std::string &error) {

    std::cout << testname << std::endl << source << ":" << lineno << std::endl
        << error << std::endl << std::endl;
    return 1;
}

