#include <libtensor/symmetry/permutation_group.h>
#include "permutation_group_test.h"

namespace libtensor {


void permutation_group_test::perform() throw(libtest::test_exception) {

	test_1();
}


/**	\test Tests the C1 group
 **/
void permutation_group_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_1()";

	try {

	permutation_group<4, double> pg;
	symmetry_element_set<4, double> set(se_perm<4, double>::k_sym_type);
	pg.convert(set);
	if(!set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "!set.is_empty()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

