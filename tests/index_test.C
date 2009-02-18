#include "index_test.h"

namespace libtensor {

void index_test::perform() throw(libtest::test_exception) {
	test_ctor();
}

void index_test::test_ctor() throw(libtest::test_exception) {
	index i1(2);
}

} // namespace libtensor

