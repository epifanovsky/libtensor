#include <libtensor.h>
#include "permutation_builder_test.h"

namespace libtensor {

void permutation_builder_test::perform() throw(libtest::test_exception) {

	char seq_ab[] = { 'a', 'b' };
	char seq_ba[] = { 'b', 'a' };

	permutation_builder<2> pb1(seq_ab, seq_ab);
	permutation<2> p1;
	if(!pb1.get_perm().equals(p1)) {
		fail_test("permutation_builder_test::perform()", __FILE__,
			__LINE__, "Test (ab, ab) failed");
	}

	permutation_builder<2> pb2(seq_ab, seq_ba);
	permutation<2> p2; p2.permute(0, 1);
	if(!pb2.get_perm().equals(p2)) {
		fail_test("permutation_builder_test::perform()", __FILE__,
			__LINE__, "Test (ab, ba) failed");
	}

}

} // namespace libtensor
