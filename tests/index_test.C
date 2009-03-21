#include "index_test.h"

namespace libtensor {

void index_test::perform() throw(libtest::test_exception) {
	test_ctor();
	test_less();
}

void index_test::test_ctor() throw(libtest::test_exception) {
	index<2> i1;
}

void index_test::test_less() throw(libtest::test_exception) {
	index<2> i1, i2;

	i1[0] = 1; i1[1] = 1;
	i2[0] = 2; i2[1] = 2;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less doesn't return (1,1)<(2,2)");
	}
	if(i2.less(i1)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(1,1)");
	}
	i1[0] = 2;
	if(!i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less doesn't return (2,1)<(2,2)");
	}
	if(i2.less(i1)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(2,1)");
	}
	i1[1] = 2;
	if(i1.less(i2)) {
		fail_test("index_test::test_less()", __FILE__, __LINE__,
			"less returns (2,2)<(2,2)");
	}
}

} // namespace libtensor

