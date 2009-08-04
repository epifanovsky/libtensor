#include <libtensor.h>
#include "so_copy_test.h"

namespace libtensor {

void so_copy_test::perform() throw(libtest::test_exception) {

	test_1();
}

void so_copy_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_copy_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 3; i2[2] = 4; i2[3] = 5;
	dimensions<4> dims(index_range<4>(i1, i2));
	default_symmetry<4, int> defsym(dims);

	so_copy<4, int> cp(defsym);
	symmetry_i<4, int> &sym = cp.get_symmetry();

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
