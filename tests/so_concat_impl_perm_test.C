#include <libtensor/symmetry/so_concat_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "so_concat_impl_perm_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_concat_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


/**	\test Tests that a concatenation of two empty group yields an empty group
		of a higher order
 **/
void so_concat_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_1()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<3, double> se3_t;
	typedef so_concat<2, 3, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<5> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
	block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
	mask<5> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<5, double> set3(se2_t::k_sym_type);
	symmetry_element_set<5, double> set3_ref(se2_t::k_sym_type);

	symmetry_operation_params< so_concat<2, 3, double> > params(
		set1, set2, permutation<5>(), bis, set3);

	so_concat_impl_t().perform(params);


	compare_ref<5>::compare(testname, bis, set3, set3_ref);

	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Concatenates a group with one element [01->10] and an empty group
 	 	(2-space) forming a 4-space. The result is expected to contain
		a single element [0123->1023]. The permutation is symmetric.
 **/
void so_concat_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_2()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
 	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	se2_t elem10(permutation<2>().permute(0, 1), true);
	se4_t elem1023(permutation<4>().permute(0, 1), true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem10);
	set3_ref.insert(elem1023);

	symmetry_operation_params< so_concat<2, 2, double> > params(
		set1, set2, permutation<4>(), bis, set3);

	so_concat_impl_t().perform(params);

	compare_ref<4>::compare(testname, bis, set3, set3_ref);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
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


/**	\test Concatenates a group with one element [01->10] and an empty group
 	 	(2-space) forming a 4-space with permutation [0123->1203]. The result
 	 	is expected to contain a single element [0123->2103]. The permutation
 	 	is symmetric.
 **/
void so_concat_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_3()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	se2_t elem1(permutation<2>().permute(0, 1), true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 1).permute(1, 2), bis, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(0, 2);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
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


/**	\test Concatenates an empty group (2-space) and a group with one element
 	 	[01->10] forming a 4-space with permutation [0123->1203]. The result
 	 	is expected to contain a single element [0123->0321]. The permutation
 	 	is symmetric.
 **/
void so_concat_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_4()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<4, double> se4_t;
	typedef so_concat<2, 2, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se2_t>
		so_concat_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
 	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	se2_t elem1(permutation<2>().permute(0, 1), true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set2.insert(elem1);

	symmetry_operation_params< so_concat<2, 2, double> > params(set1,
			set2, permutation<4>().permute(0, 1).permute(1, 2), bis, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(1, 3);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
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


/**	\test Concatenates an empty group (1-space) and a group with one element
 	 	[012->120] forming a 4-space with permutation [0123->3102]. The result
 	 	is expected to contain a single element [0123->1320]. The permutation
 	 	is symmetric.
 **/
void so_concat_impl_perm_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_5()";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;
	typedef so_concat<1, 3, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se1_t>
		so_concat_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
 	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	se3_t elem1(permutation<3>().permute(0, 1).permute(1, 2), true);

	symmetry_element_set<1, double> set1(se1_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set2.insert(elem1);

	permutation<4> perm;
	perm.permute(2, 3).permute(0, 2);
	symmetry_operation_params< so_concat<1, 3, double> > params(
		set1, set2, perm, bis, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(0, 1).permute(1, 3);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();

	const se4_t &elem2 = adapter.get_elem(i);
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


/**	\test Concatenates a group with one element [012->120] and an empty group
 	 	(1-space) forming a 4-space with permutation [0123->3102]. The result
 	 	is expected to contain a single element [0123->0312]. The permutation
 	 	is antisymmetric.
 **/
void so_concat_impl_perm_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_concat_impl_perm_test::test_6()";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<3, double> se3_t;
	typedef se_perm<4, double> se4_t;
	typedef so_concat<3, 1, double> so_concat_t;
	typedef symmetry_operation_impl<so_concat_t, se3_t>
		so_concat_impl_t;

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 1);
	bis.split(m, 2);

	se3_t elem1(permutation<3>().permute(0, 1).permute(1, 2), false);

	symmetry_element_set<3, double> set1(se3_t::k_sym_type);
	symmetry_element_set<1, double> set2(se1_t::k_sym_type);
	symmetry_element_set<4, double> set3(se4_t::k_sym_type);
	symmetry_element_set<4, double> set3_ref(se4_t::k_sym_type);

	set1.insert(elem1);

	permutation<4> perm; perm.permute(2, 3).permute(0, 2);
	symmetry_operation_params< so_concat<3, 1, double> > params(
		set1, set2, perm, bis, set3);

	so_concat_impl_t().perform(params);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<4> p2; p2.permute(2, 3).permute(1, 2);
	symmetry_element_set_adapter<4, double, se4_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se4_t>::iterator i =
		adapter.begin();
	const se4_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "elem2.is_symm()");
	}
	if(!elem2.get_perm().equals(p2)) {
		fail_test(testname, __FILE__, __LINE__, "elem2.perm != p2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}



} // namespace libtensor
