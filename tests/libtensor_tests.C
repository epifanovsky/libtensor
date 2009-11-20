#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include "libtensor_suite.h"

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

	srand48(time(NULL));

	ostringstream ss;
	ss << " Unit tests for libtensor "
		<< version::get_string() << " ";
	string separator(ss.str().size(), '-');
	cout << separator << endl << ss.str() << endl << separator << endl;

	suite_handler handler;
	libtensor_suite suite;
	suite.set_handler(&handler);

	if(argc == 1) {
		suite.run_all_tests();
	} else {
		for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
	}
}

