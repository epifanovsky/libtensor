#include <symmetry/so_union_impl_perm.h>
#include <btod/transf_double.h>
#include "so_union_impl_perm_test.h"

namespace libtensor {


void so_union_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
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

	if(!set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests that the union of an empty set and a non-empty set yields
		the non-empty set
 **/
void so_union_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_union_impl_perm_test::test_2()";

	typedef se_perm<2, double> se_t;

	try {

	permutation<2> p; p.permute(0, 1);
	se_perm<2, double> elem1(p, true);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);

	symmetry_operation_params< so_union<2, double> > params(set1, set2);

	so_union_impl< se_perm<2, double> > op;
	op.perform(params, set3);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
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


/**	\test Tests that the union of two non-overlapping non-empty sets
		combines the sets
 **/
void so_union_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_union_impl_perm_test::test_3()";

	typedef se_perm<4, double> se_t;

	try {

	permutation<4> p1, p2;
	p1.permute(0, 2);
	p2.permute(1, 3);
	se_perm<4, double> elem1(p1, true), elem2(p2, true);

	symmetry_element_set<4, double> set1(se_t::k_sym_type);
	symmetry_element_set<4, double> set2(se_t::k_sym_type);
	symmetry_element_set<4, double> set3(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem2);

	symmetry_operation_params< so_union<4, double> > params(set1, set2);

	so_union_impl< se_perm<4, double> > op;
	op.perform(params, set3);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	symmetry_element_set_adapter<4, double, se_t> adapter(set3);
	symmetry_element_set_adapter<4, double, se_t>::iterator i =
		adapter.begin();
	const se_t &elem1a = adapter.get_elem(i);
	i++;
	if(i == adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected two elements in the result (1).");
	}
	const se_t &elem2a = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected two elements in the result (2).");
	}

	if(!elem1a.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem1a.is_symm()");
	}
	if(!elem2a.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!elem2a.is_symm()");
	}
	if(!elem1a.get_perm().equals(elem1.get_perm())) {
		if(!elem1a.get_perm().equals(elem2.get_perm())) {
			fail_test(testname, __FILE__, __LINE__,
				"Bad elem1a.");
		}
		if(!elem2a.get_perm().equals(elem1.get_perm())) {
			fail_test(testname, __FILE__, __LINE__,
				"elem1a == elem2, elem2a != elem1");
		}
	} else {
		if(!elem2a.get_perm().equals(elem2.get_perm())) {
			fail_test(testname, __FILE__, __LINE__,
				"elem1a == elem1, elem2a != elem2");
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the union of two identical non-empty set
 **/
void so_union_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_union_impl_perm_test::test_4()";

	typedef se_perm<2, double> se_t;

	try {

	permutation<2> p; p.permute(0, 1);
	se_perm<2, double> elem1(p, true);

	symmetry_element_set<2, double> set1(se_t::k_sym_type);
	symmetry_element_set<2, double> set2(se_t::k_sym_type);
	symmetry_element_set<2, double> set3(se_t::k_sym_type);

	set1.insert(elem1);
	set2.insert(elem1);

	symmetry_operation_params< so_union<2, double> > params(set1, set2);

	so_union_impl< se_perm<2, double> > op;
	op.perform(params, set3);

	if(set3.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
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

