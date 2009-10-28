#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "libtensor_suite.h"

using namespace libtensor;
using libtest::test_exception;

class suite_handler : public libtest::suite_event_handler {
public:
	virtual void on_suite_start(const char *suite) {
	}

	virtual void on_suite_end(const char *suite) {
	}

	virtual void on_test_start(const char *test) {
		printf("Test %s ... ", test); fflush(stdout);
	}

	virtual void on_test_end_success(const char *test) {
		printf("done.\n"); fflush(stdout);
	}

	virtual void on_test_end_exception(const char *test,
		const test_exception &e) {
		printf("FAIL!\n");
		printf("%s\n", e.what());
		fflush(stdout);
	}
};

int main(int argc, char **argv) {

	srand48(time(NULL));

	char smsg[81], sline[81];
	snprintf(smsg, 81, "Performing tests for libtensor revision %s",
		libtensor::version);
	size_t slen = strlen(smsg);
	memset(sline, '-', 80);
	sline[slen] = '\0';
	puts(sline); puts(smsg); puts(sline);

	suite_handler handler;
	libtensor_suite suite;
	suite.set_handler(&handler);

	if(argc == 1) {
		suite.run_all_tests();
	} else {
		for(int i = 1; i < argc; i++) suite.run_test(argv[i]);
	}
}

