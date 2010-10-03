#include <libtensor/symmetry/so_merge_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "so_merge_impl_perm_test.h"

namespace libtensor {


void so_merge_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Tests that a merge of 2 dims of an empty group yields an empty group
		of a lower order
 **/
void so_merge_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_perm_test::test_1()";

	typedef se_perm<4, double> se4_t;
	typedef se_perm<3, double> se3_t;
	typedef so_merge<4, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	mask<4> msk; msk[0] = true; msk[1] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 2 dims of one 2-cycle in a 2-space onto a 1-space. Expected
		result: C1 in 1-space. Symmetric elements.
 **/
void so_merge_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_perm_test::test_2()";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<2, double> se2_t;
	typedef so_merge<2, 2, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se2_t>
		so_merge_impl_t;

	try {

	se2_t elem1(permutation<2>().permute(0, 1), true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<1, double> set2(se1_t::k_sym_type);

	set1.insert(elem1);

	mask<2> msk; msk[0] = true; msk[1] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 3 dims of a 2-cycle in a 5-space onto a 3-space untouched by
		the masks. Expected result: 2-cycle in 3-space.
		Symmetric elements.
 **/
void so_merge_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_perm_test::test_3()";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<5, double> se5_t;
	typedef so_merge<5, 3, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se5_t>
		so_merge_impl_t;

	try {

	se5_t elem1(permutation<5>().permute(0, 1), true);

	symmetry_element_set<5, double> set1(se5_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	set1.insert(elem1);

	mask<5> msk; msk[2] = true; msk[3] = true; msk[4] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<3> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 3 dim of a 3-cycle in a 6-space onto a 4-space with one
		dimension out. Expected result: C1 in 4-space.
		Symmetric elements.
 **/
void so_merge_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_perm_test::test_4()";

	typedef se_perm<6, double> se6_t;
	typedef se_perm<4, double> se4_t;
	typedef so_merge<6, 3, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se6_t>
		so_merge_impl_t;

	try {

	se6_t elem1(permutation<6>().permute(0, 1).permute(1, 3), true);

	symmetry_element_set<6, double> set1(se6_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	set1.insert(elem1);

	mask<6> msk;
	msk[3] = true; msk[4] = true; msk[5] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
