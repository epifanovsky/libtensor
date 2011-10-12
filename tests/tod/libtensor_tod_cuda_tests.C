#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <libtensor/version.h>
#include <libtensor/mp/worker_pool.h>
#include "libtensor_tod_cuda_suite.h"

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

	srand48(time(0));

//	unsigned ngroups[4] = { 0, 1, 2, 5 };

	unsigned ngroups[1] = { 0 };

	for(size_t igrp = 0; igrp < 1; igrp++) {
	for(size_t nthr = 1; nthr <= 1; nthr++) {

		ostringstream ss1, ss2;
		ss1 << " Unit tests for libtensor " << version::get_string() << " ";
		if(ngroups[igrp] == 0) {
			ss1 << "(single-threaded) ";
		} else {
			ss1 << "(multi-threaded, "
				<< ngroups[igrp] + 1 << "/" << nthr
				<< " threads) ";
		}
		ss2 << " Tensor operations (double) test suite. ";
		string separator(std::max(ss1.str().size(), ss2.str().size()), '-');
		cout << separator << endl << ss1.str() << endl << ss2.str() << endl
			<< separator << endl;

		if(ngroups[igrp] > 0) {
			libtensor::worker_pool::get_instance().
				init(ngroups[igrp], nthr);
		}

		suite_handler handler;
		libtensor_tod_cuda_suite suite;
		suite.set_handler(&handler);

		if(argc == 1) {
			suite.run_all_tests();
		} else {
			for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
		}

		if(ngroups[igrp] > 0) {
			libtensor::worker_pool::get_instance().shutdown();
		}
	}
}
}

