#include <symmetry/so_union_impl_perm.h>
#include "so_union_impl_perm_test.h"

namespace libtensor {


void so_union_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
}


/**	\test Tests that the union of two empty sets yields an empty set
 **/
void so_union_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_union_impl_perm_test::test_1()";

	typedef se_perm<2, double> se_t;

	try {

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	symmetry_operation_params< so_union<2, double> > params(set1, set2);

	so_union_impl< se_perm<2, double> > op;
	op.perform(params, set3);

	if(set3.begin() != set3.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

