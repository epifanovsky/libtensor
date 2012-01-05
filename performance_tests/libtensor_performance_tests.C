#include <sstream>
#include <iostream>
#include "libtensor_pt_suite.h"

using namespace libtensor;
using namespace std;
using libtest::test_exception;

class performance_suite_handler
	: public libtest::suite_event_handler
{
public:
	virtual void on_suite_start(const char *suite) {
	}

	virtual void on_suite_end(const char *suite) {
	}

	virtual void on_test_start(const char *test) {
		cout << "Performance test " << test << endl;
		cout.flush();

		// reset timings
		libutil::global_timings::get_instance().reset();
	}

	virtual void on_test_end_success(const char *test) {
		cout << " ... Test done." << endl;
		// print timings
		if ( libutil::global_timings::get_instance().ntimings() > 0 ) {
			cout << "Timings are: " << endl;
			cout << libutil::global_timings::get_instance() << endl;
		}
		else
			cout << "No Timings" << endl;
		cout.flush();
	}

	virtual void on_test_end_exception(const char *test,
		const test_exception &e) {
		cout << " ... FAIL!" << endl;
		cout << e.what() << endl;
		cout.flush();
	}
};

int main(int argc, char **argv) {

	ostringstream ss;
	ss << " Performance tests for libtensor "
		<< version::get_string() << " ";
	string separator(ss.str().size(), '-');
	cout << separator << endl << ss.str() << endl << separator << endl;

	performance_suite_handler handler;
	libtensor_pt_suite suite;
	suite.set_handler(&handler);
	return suite.run_all_tests();
}
