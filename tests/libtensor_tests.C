#include "libtensor_suite.h"
#include <cstdio>

using namespace libtensor;
using libtest::test_exception;

int main(int argc, char **argv) {
	libtensor_suite suite;
	try {
		suite.run_all_tests();
	} catch(test_exception e) {
		printf("%s\n", e.what());
	}
}

