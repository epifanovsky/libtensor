#include "lehmer_code_test.h"

namespace libtensor {

void lehmer_code_test::perform() throw(libtest::test_exception) {
	test_code(2);
	test_code(3);
	test_code(4);
	test_code(5);
	test_code(6);
}

void lehmer_code_test::test_code(const size_t order)
	throw(libtest::test_exception) {
}

} // namespace libtensor

