#include <libtensor/core/permutation_builder.h>
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

	char seq_abcd[] = { 'a', 'b', 'c', 'd' };
	char seq_acdb[] = { 'a', 'c', 'd', 'b' };
	char seq_abcd_1[4];
	permutation_builder<4> pb3(seq_abcd, seq_acdb);
	permutation<4> p3; p3.permute(1, 2).permute(1, 3);
	for(size_t i = 0; i < 4; i++) seq_abcd_1[i] = seq_acdb[i];
	p3.apply(seq_abcd_1);
	for(size_t i = 0; i < 4; i++) {
		if(seq_abcd_1[i] != seq_abcd[i]) {
			fail_test("permutation_builder_test::perform()",
				__FILE__, __LINE__,
				"Wrong reference in (abcd, acdb).");
		}
	}
	if(!pb3.get_perm().equals(p3)) {
		fail_test("permutation_builder_test::perform()", __FILE__,
			__LINE__, "Test (abcd, acdb) failed");
	}

}

} // namespace libtensor
