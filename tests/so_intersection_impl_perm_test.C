#include <libtensor/symmetry/so_intersection_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "so_intersection_impl_perm_test.h"

namespace libtensor {


void so_intersection_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


/**	\test Tests that the intersection of two empty sets yields an empty set
 **/
void so_intersection_impl_perm_test::test_1() throw(libtest::test_exception) {

	static const char *testname =
		"so_intersection_impl_perm_test::test_1()";

	typedef se_perm<2, double> se_t;
	typedef so_intersection<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;

	try {

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	params_t params(set1, set2);

	so_intersection_impl<se_t> op;
	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	permutation<2> perm; perm.permute(0, 1);
	se_t elem(perm, true);
	set3.insert(elem);

	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (2).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests that the intersection of an empty set and a non-empty set
		yields an empty set
 **/
void so_intersection_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname =
		"so_intersection_impl_perm_test::test_2()";

	typedef se_perm<2, double> se_t;
	typedef so_intersection<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;

	try {

	permutation<2> p; p.permute(0, 1);
	se_t elem1(p, true);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);

	params_t params(set1, set2);

	so_intersection_impl<se_t> op;
	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	set3.insert(elem1);
	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests that the intersection of two non-overlapping non-empty sets
		yields an empty set
 **/
void so_intersection_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname =
		"so_intersection_impl_perm_test::test_3()";

	typedef se_perm<4, double> se_t;
	typedef so_intersection<4, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;

	try {

	permutation<4> p1, p2;
	p1.permute(0, 2);
	p2.permute(1, 3);
	se_t elem1(p1, true), elem2(p2, true);

	symmetry_element_set<4, double> set1(se_t::k_sym_type);
	symmetry_element_set<4, double> set2(se_t::k_sym_type);
	symmetry_element_set<4, double> set3(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem2);

	params_t params(set1, set2);

	so_intersection_impl<se_t> op;
	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (1).");
	}

	set3.insert(elem1);
	set3.insert(elem2);
	op.perform(params, set3);
	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"!set3.is_empty() (2).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the intersection of two identical non-empty sets
 **/
void so_intersection_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname =
		"so_intersection_impl_perm_test::test_4()";

	typedef se_perm<2, double> se_t;
	typedef so_intersection<2, double> so_t;
	typedef symmetry_operation_params<so_t> params_t;

	try {

	permutation<2> p; p.permute(0, 1);
	se_t elem1(p, true);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem1);

	params_t params(set1, set2);

	so_intersection_impl<se_t> op;
	op.perform(params, set3);
	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "set3.is_empty()");
	}

	symmetry_element_set_adapter<2, double, se_t> adapter(set3);
	symmetry_element_set_adapter<2, double, se_t>::iterator i =
		adapter.begin();
	const se_t &elem2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(!elem2.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2.is_symm()");
	}
	if(!elem1.get_perm().equals(elem2.get_perm())) {
		fail_test(testname, __FILE__, __LINE__, "elem1 != elem2");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

