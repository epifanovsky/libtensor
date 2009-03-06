#include "libtensor_suite.h"
#include <cstdio>
#include <cstring>

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
	suite.run_all_tests();
}

