#include "dimensions_test.h"

namespace libtensor {

void dimensions_test::perform() throw(libtest::test_exception) {
	test_ctor();
}

void dimensions_test::test_ctor() throw(libtest::test_exception) {
	index i1a(2), i1b(2);
	i1b[0] = 2; i1b[1] = 3;
	index_range ir1(i1a, i1b);
	dimensions d1(ir1);

	if(d1.get_order() != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of dimensions in d1");
	}
	if(d1[0] != 2) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d1[0]");
	}
	if(d1[1] != 3) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect number of elements along d1[1]");
	}
	if(d1.get_size() != 6) {
		fail_test("dimensions_test::test_ctor()", __FILE__, __LINE__,
			"Incorrect total number of elements in d1");
	}
}

} // namespace libtensor

