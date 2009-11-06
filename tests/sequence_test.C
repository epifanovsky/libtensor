#include <libtensor.h>
#include "sequence_test.h"

namespace libtensor {


void sequence_test::perform() throw(libtest::test_exception) {

	test_1();
}


void sequence_test::test_1() throw(libtest::test_exception) {

	//	Tests sequence<0>

	static const char *testname = "sequence_test::test_1()";

	try {

	sequence<0, int> seq(0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
