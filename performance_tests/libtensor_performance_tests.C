#include "libtensor_performance_suite.h"
#include "../core/global_timings.h"

#include <iostream>

using namespace libtensor;
using libtest::test_exception;

class performance_suite_handler : public libtest::suite_event_handler {
public:
	virtual void on_suite_start(const char *suite) {
	}

	virtual void on_suite_end(const char *suite) {
	}

	virtual void on_test_start(const char *test) {
		std::cout << "Performance test " << test << " ... ";
		std::cout.flush();
		
		// reset timings
		global_timings::get_instance().reset();
		
	}

	virtual void on_test_end_success(const char *test) {
		std::cout << "done." << std::endl; 
		// print timings
		std::cout << "Timings are: " << std::endl;
		std::cout << global_timings::get_instance() << std::endl;
		std::cout.flush();
	}

	virtual void on_test_end_exception(const char *test,
		const test_exception &e) {
		std::cout << "FAIL!" << std::endl;
		std::cout << e.what() << std::endl;
		std::cout.flush();
	}
};

int main(int argc, char **argv) {
	char smsg[81], sline[81];
	snprintf(smsg, 81, "Performing performance tests for libtensor revision %s",
		libtensor::version);
	size_t slen = strlen(smsg);
	memset(sline, '-', 80);
	sline[slen] = '\0';
	puts(sline); puts(smsg); puts(sline);

	performance_suite_handler handler;
	libtensor_performance_suite suite;
	suite.set_handler(&handler);
	suite.run_all_tests();
}

