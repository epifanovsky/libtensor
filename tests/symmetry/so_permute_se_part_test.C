#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/so_permute_se_part.h>
#include "../compare_ref.h"
#include "so_permute_se_part_test.h"

namespace libtensor {

void so_permute_se_part_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2a();
	test_2b();
	test_3();

}


/**	\test Tests permutation of an empty set
 **/
void so_permute_se_part_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_permute_se_part_test::test_1()";

	typedef se_part<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_se_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 7;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis4.split(m4, 2);
	bis4.split(m4, 4);
	bis4.split(m4, 6);

	index<4> i0000, i0011, i1100, i0110, i1001, i0101, i1010, i1111;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_se_t().perform(params);

	if(! set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests permutation of a non-empty set. Permutation does not affect
		the mapping
 **/
void so_permute_se_part_test::test_2a() throw(libtest::test_exception) {

	static const char *testname = "so_permute_se_part_test::test_2a()";

	typedef se_part<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_se_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 7;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m1111, m1100;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	m1100[0] = true; m1100[1] = true;
	bis4.split(m1111, 2);
	bis4.split(m1111, 4);
	bis4.split(m1111, 6);

	index<4> i0000, i0001, i0010, i0100, i1000,
		i0011, i0101, i0110, i1001, i1010, i1100,
		i0111, i1011, i1101, i1110, i1111;
	i1000[0] = 1; i0100[1] = 1; i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1; i0101[1] = 1; i0101[3] = 1;
	i0110[1] = 1; i0110[2] = 1; i1001[0] = 1; i1001[3] = 1;
	i1010[0] = 1; i1010[2] = 1; i1100[0] = 1; i1100[1] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1011[0] = 1; i1011[2] = 1; i1011[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i1101[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	scalar_transf<double> tr0;

	se4_t elem(bis4, m1100, 2);
	elem.add_map(i0000, i1100, tr0);
	elem.add_map(i0100, i1000, tr0);

	permutation<4> perm;
	perm.permute(2, 3);
	bis4.permute(perm);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_se_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests permutation of a non-empty set. Permutation does not affect
		the mapping
 **/
void so_permute_se_part_test::test_2b() throw(libtest::test_exception) {

	static const char *testname = "so_permute_se_part_test::test_2b()";

	typedef se_part<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_se_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 7;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m1111;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis4.split(m1111, 2);
	bis4.split(m1111, 4);
	bis4.split(m1111, 6);

	index<4> i0000, i0001, i0010, i0100, i1000,
		i0011, i0101, i0110, i1001, i1010, i1100,
		i0111, i1011, i1101, i1110, i1111;
	i1000[0] = 1; i0100[1] = 1; i0010[2] = 1; i0001[3] = 1;
	i0011[2] = 1; i0011[3] = 1; i0101[1] = 1; i0101[3] = 1;
	i0110[1] = 1; i0110[2] = 1; i1001[0] = 1; i1001[3] = 1;
	i1010[0] = 1; i1010[2] = 1; i1100[0] = 1; i1100[1] = 1;
	i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
	i1011[0] = 1; i1011[2] = 1; i1011[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i1101[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	scalar_transf<double> tr0;

	se4_t elem(bis4, m1111, 2);
	elem.add_map(i0000, i1100, tr0);
	elem.add_map(i0100, i1000, tr0);
	elem.add_map(i0011, i1111, tr0);
	elem.add_map(i0111, i1011, tr0);

	permutation<4> perm;
	perm.permute(2, 3);
	bis4.permute(perm);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_se_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests permutation of a non-empty set.
 **/
void so_permute_se_part_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_permute_se_part_test::test_3()";

	typedef se_part<4, double> se4_t;
	typedef so_permute<4, double> so_permute_t;
	typedef symmetry_operation_impl<so_permute_t, se4_t>
		so_permute_se_t;

	try {

	index<4> i4a, i4b;
	i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 7;

	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);

	index<4> i0000, i0011, i1100, i0110, i1001, i0101, i1010, i1111;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	scalar_transf<double> tr0;

	se4_t elem(bis4, m4, 2);
	elem.add_map(i0000, i1111, tr0);
	elem.add_map(i0011, i1100, tr0);
	elem.add_map(i0110, i1001, tr0);

	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	bis4.permute(perm);

	se4_t elem_ref(bis4, m4, 2);
	elem_ref.add_map(i0000, i1111, tr0);
	elem_ref.add_map(i0101, i1010, tr0);
	elem_ref.add_map(i1100, i0011, tr0);

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);
	symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

	set1.insert(elem);
	set2_ref.insert(elem_ref);

	symmetry_operation_params<so_permute_t> params(set1, perm, set2);

	so_permute_se_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
	}

	compare_ref<4>::compare(testname, bis4, set2, set2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}





} // namespace libtensor
