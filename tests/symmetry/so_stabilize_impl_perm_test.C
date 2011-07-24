#include <libtensor/symmetry/perm/so_stabilize_impl_perm.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_stabilize_impl_perm_test.h"

namespace libtensor {


void so_stabilize_impl_perm_test::perform() throw(libtest::test_exception) {

	test_1a();
	test_1b();
	test_2();
	test_3();
	test_4();
    test_5();
}


/**	\test Tests that a projection of an empty group yields an empty group
		of a lower order
 **/
void so_stabilize_impl_perm_test::test_1a() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_perm_test::test_1a()";

	typedef se_perm<4, double> se4_t;
	typedef se_perm<2, double> se2_t;
	typedef so_stabilize<4, 2, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<4> msk[1]; msk[0][0] = true; msk[0][1] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Tests that a double projection of an empty group yields an empty
		group of a lower order
 **/
void so_stabilize_impl_perm_test::test_1b() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_perm_test::test_1b()";

	typedef se_perm<4, double> se4_t;
	typedef se_perm<2, double> se2_t;
	typedef so_stabilize<4, 2, 2, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se4_t>
		so_stabilize_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	mask<4> msk[2]; msk[0][0] = true; msk[1][1] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of one 2-cycle in a 2-space onto a 1-space. Expected
		result: C1 in 1-space. Symmetric elements.
 **/
void so_stabilize_impl_perm_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_perm_test::test_2()";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<2, double> se2_t;
	typedef so_stabilize<2, 1, 1, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se2_t>
		so_stabilize_impl_t;

	try {

	permutation<2> p1; p1.permute(0, 1);
	se2_t elem1(p1, true);

	symmetry_element_set<2, double> set1(se2_t::k_sym_type);
	symmetry_element_set<1, double> set2(se1_t::k_sym_type);

	set1.insert(elem1);

	mask<2> msk[1]; msk[0][0] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Projection of a 2-cycle in a 5-space onto a 2-space untouched by
		the masks. Expected result: 2-cycle in 2-space.
		Symmetric elements.
 **/
void so_stabilize_impl_perm_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_perm_test::test_3()";

	typedef se_perm<2, double> se2_t;
	typedef se_perm<5, double> se5_t;
	typedef so_stabilize<5, 3, 2, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se5_t>
		so_stabilize_impl_t;

	try {

	permutation<5> p1; p1.permute(0, 1);
	se5_t elem1(p1, true);

	symmetry_element_set<5, double> set1(se5_t::k_sym_type);
	symmetry_element_set<2, double> set2(se2_t::k_sym_type);

	set1.insert(elem1);

	mask<5> msk[2]; msk[0][2] = true; msk[0][3] = true; msk[1][4] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<2> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
	symmetry_element_set_adapter<2, double, se2_t>::iterator i =
		adapter.begin();
	const se2_t &elem2 = adapter.get_elem(i);
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


/**	\test Projection of a 3-cycle in a 6-space onto a 3-space with one
		dimension out. Expected result: C1 in 3-space.
		Symmetric elements.
 **/
void so_stabilize_impl_perm_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_stabilize_impl_perm_test::test_4()";

	typedef se_perm<6, double> se6_t;
	typedef se_perm<3, double> se3_t;
	typedef so_stabilize<6, 3, 3, double> so_stabilize_t;
	typedef symmetry_operation_impl<so_stabilize_t, se6_t>
		so_stabilize_impl_t;

	try {

	permutation<6> p1; p1.permute(0, 1).permute(1, 3);
	se6_t elem1(p1, true);

	symmetry_element_set<6, double> set1(se6_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	set1.insert(elem1);

	mask<6> msk[3];
	msk[0][5] = true; msk[1][3] = true; msk[2][4] = true;
	symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

	so_stabilize_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/** \test
 **/
void so_stabilize_impl_perm_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "so_stabilize_impl_perm_test::test_5()";

    typedef se_perm<8, double> se8_t;
    typedef se_perm<4, double> se4_t;
    typedef so_stabilize<8, 4, 2, double> so_stabilize_t;
    typedef symmetry_operation_impl<so_stabilize_t, se8_t>
        so_stabilize_impl_t;

    try {

    permutation<8> p1, p2;
    p1.permute(0, 1).permute(2, 3);
    p2.permute(4, 5).permute(6, 7);
    se8_t elem1(p1, true), elem2(p2, true);

    permutation<4> p3;
    p3.permute(0, 1).permute(2, 3);
    se4_t elem3(p3, true);

    symmetry_element_set<8, double> set1(se8_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

    set1.insert(elem1);
    set1.insert(elem2);
    set2_ref.insert(elem3);

    mask<8> msk[2];
    msk[0][2] = true; msk[0][6] = true; msk[1][3] = true; msk[1][7] = true;
    symmetry_operation_params<so_stabilize_t> params(set1, msk, set2);

    so_stabilize_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    index<4> i1, i2;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
    mask<4> m;
    m[0] = m[1] = m[2] = m[3] = true;
    bis.split(m, 1);
    bis.split(m, 2);
    bis.split(m, 3);

    compare_ref<4>::compare(testname, bis, set2, set2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
