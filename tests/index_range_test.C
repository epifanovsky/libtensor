#include "index_range_test.h"

namespace libtensor {

void index_range_test::perform() throw(libtest::test_exception) {
	test_ctor();
}

void index_range_test::test_ctor() throw(libtest::test_exception) {
	index i1(2), i2(2);
	i2[0] = 2; i2[1] = 2;
	index_range(i1, i2);
}

} // namespace libtensor

