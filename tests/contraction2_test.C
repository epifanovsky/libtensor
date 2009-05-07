#include "contraction2_test.h"

namespace libtensor {

void contraction2_test::perform() throw(libtest::test_exception) {
	permutation<4> perm;
	contraction2<2,2,2> c(perm);

	if(c.is_complete()) {
		fail_test("contraction2_test::perform()", __FILE__, __LINE__,
			"Empty contraction declares complete");
	}

	c.contract(2, 2);
	if(c.is_complete()) {
		fail_test("contraction2_test::perform()", __FILE__, __LINE__,
			"Incomplete contraction declares complete");
	}

	c.contract(3, 3);
	if(!c.is_complete()) {
		fail_test("contraction2_test::perform()", __FILE__, __LINE__,
			"Complete contraction declares incomplete");
	}
}

} // namespace libtensor
