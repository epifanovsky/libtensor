#include <libtensor/symmetry/so_merge_se_perm.h>
#include <sstream>
#include "so_merge_impl_perm_test.h"

namespace libtensor {


void so_merge_impl_perm_test::perform() throw(libtest::test_exception) {

	test_empty_1();
    test_empty_2();
    test_empty_3();
	test_nn1(true); test_nn1(false);
    test_nm1_1(true); test_nm1_1(false);
    test_nm1_2(true); test_nm1_2(false);
    test_nm1_3(true); test_nm1_3(false);
    test_2n2nn_1(true, true);  test_2n2nn_1(true, false);
    test_2n2nn_1(false, true); test_2n2nn_1(false, false);
    test_2n2nn_2(true); test_2n2nn_2(false);
    test_nmk_1(true); test_nmk_1(false);
    test_nmk_2(true); test_nmk_2(false);
}


/**	\test Tests that a merge of 2 dims of an empty group yields an empty group
		of a lower order
 **/
void so_merge_impl_perm_test::test_empty_1() throw(libtest::test_exception) {

	static const char *testname = "so_merge_impl_perm_test::test_empty_1()";

	typedef se_perm<4, double> se4_t;
	typedef se_perm<3, double> se3_t;
	typedef so_merge<4, 2, 1, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se4_t>
		so_merge_impl_t;

	try {

	symmetry_element_set<4, double> set1(se4_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	mask<4> msk[1]; msk[0][0] = true; msk[0][1] = true;
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

/** \test Tests that a merge of all dims of an empty group yields an empty
        group of dim 1
 **/
void so_merge_impl_perm_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_merge_impl_perm_test::test_empty_2()";

    typedef se_perm<3, double> se3_t;
    typedef se_perm<1, double> se1_t;
    typedef so_merge<3, 3, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t>
        so_merge_impl_t;

    try {

    symmetry_element_set<3, double> set1(se3_t::k_sym_type);
    symmetry_element_set<1, double> set2(se1_t::k_sym_type);

    mask<3> msk[1]; msk[0][0] = true; msk[0][1] = true; msk[0][2] = true;
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

/** \test Tests that multiple merges of 2 dims of an empty group yield an
        empty group of lower order
 **/
void so_merge_impl_perm_test::test_empty_3() throw(libtest::test_exception) {

    static const char *testname = "so_merge_impl_perm_test::test_empty_3()";

    typedef se_perm<6, double> se6_t;
    typedef se_perm<3, double> se3_t;
    typedef so_merge<6, 5, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se6_t>
        so_merge_impl_t;

    try {

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<3, double> set2(se3_t::k_sym_type);

    mask<6> msk[2];
    msk[0][0] = true; msk[0][1] = true;
    msk[1][3] = true; msk[1][4] = true; msk[1][5] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Merge of 2 dims of one 2-cycle in a 2-space onto a 1-space. Expected
		result: C1 in 1-space. Symmetric elements.
 **/
void so_merge_impl_perm_test::test_nn1(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_nn1(" << symm << ")";

	typedef se_perm<1, double> se1_t;
	typedef se_perm<2, double> se2_t;
	typedef so_merge<2, 2, 1, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se2_t>
		so_merge_impl_t;

	try {

	se2_t el1(permutation<2>().permute(0, 1), symm);

	symmetry_element_set<2, double> seta(se2_t::k_sym_type);
	symmetry_element_set<1, double> setb(se1_t::k_sym_type);

	seta.insert(el1);

	mask<2> msk[1]; msk[0][0] = true; msk[0][1] = true;
	symmetry_operation_params<so_merge_t> params(seta, msk, setb);

	if (symm) {

	    so_merge_impl_t().perform(params);

	    if(! setb.is_empty()) {
	        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
	                "Expected an empty set.");
	    }
	}
	else {

	    bool failed = false;
	    try {
	        so_merge_impl_t().perform(params);
	    }
	    catch (exception &e) {
	        failed = true;
	    }
	    if (! failed) {
	        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
	                "Merge did not fail");
	    }
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 3 dims of a 2-cycle in a 5-space onto a 3-space untouched by
		the masks. Expected result: 2-cycle in 3-space.
 **/
void so_merge_impl_perm_test::test_nm1_1(
        bool symm) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_merge_impl_perm_test::test_nm1_1(" << symm << ")";

	typedef se_perm<3, double> se3_t;
	typedef se_perm<5, double> se5_t;
	typedef so_merge<5, 3, 1, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se5_t>
		so_merge_impl_t;

	try {

	se5_t el1(permutation<5>().permute(0, 1), symm);

	symmetry_element_set<5, double> set1(se5_t::k_sym_type);
	symmetry_element_set<3, double> set2(se3_t::k_sym_type);

	set1.insert(el1);

	mask<5> msk[1]; msk[0][2] = true; msk[0][3] = true; msk[0][4] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"Expected a non-empty set.");
	}

	permutation<3> p2; p2.permute(0, 1);
	symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
	symmetry_element_set_adapter<3, double, se3_t>::iterator i =
		adapter.begin();
	const se3_t &el2 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"Expected only one element.");
	}
	if(el2.is_symm() != symm) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
	}
	if(!el2.get_perm().equals(p2)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
		        "el2.get_perm() != p2");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Merge of 3 dim of a 3-cycle in a 6-space onto a 4-space with one
		dimension out. Expected result: C1 in 4-space.
 **/
void so_merge_impl_perm_test::test_nm1_2(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_nm1_2(" << symm << ")";

	typedef se_perm<6, double> se6_t;
	typedef se_perm<4, double> se4_t;
	typedef so_merge<6, 3, 1, double> so_merge_t;
	typedef symmetry_operation_impl<so_merge_t, se6_t>
		so_merge_impl_t;

	try {

	se6_t el1(permutation<6>().permute(0, 1).permute(1, 3).permute(3, 4), symm);

	symmetry_element_set<6, double> set1(se6_t::k_sym_type);
	symmetry_element_set<4, double> set2(se4_t::k_sym_type);

	set1.insert(el1);

	mask<6> msk[1];
	msk[0][3] = true; msk[0][4] = true; msk[0][5] = true;
	symmetry_operation_params<so_merge_t> params(set1, msk, set2);

	so_merge_impl_t().perform(params);

	if(!set2.is_empty()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__,
			"Expected an empty set.");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

/** \test Merge of 2 dim of a group in a 4-space onto a 3-space.
 **/
void so_merge_impl_perm_test::test_nm1_3(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_nm1_3(" << symm << ")";

    typedef se_perm<4, double> se4_t;
    typedef se_perm<3, double> se3_t;
    typedef so_merge<4, 2, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
        so_merge_impl_t;

    try {

    se4_t el1(permutation<4>().permute(0, 1).permute(2, 3), symm);

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<3, double> set2(se3_t::k_sym_type);

    set1.insert(el1);

    mask<4> msk[1];
    msk[0][2] = true; msk[0][3] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    permutation<3> p2; p2.permute(0, 1);
    symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
    symmetry_element_set_adapter<3, double, se3_t>::iterator i =
        adapter.begin();
    const se3_t &el2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.is_symm() != symm) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "el2.get_perm() != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of a group in a 4-space onto a 2-space.
 **/
void so_merge_impl_perm_test::test_2n2nn_1(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_2n2nn_1(" <<
            symm1 << "," << symm2 << ")";

    typedef se_perm<4, double> se4_t;
    typedef se_perm<2, double> se2_t;
    typedef so_merge<4, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
        so_merge_impl_t;

    try {

    se4_t el1a(permutation<4>().permute(0, 1), symm1);
    se4_t el1b(permutation<4>().permute(2, 3), symm2);

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1a);
    set1.insert(el1b);

    mask<4> msk[2];
    msk[0][0] = true; msk[0][2] = true;
    msk[1][1] = true; msk[1][3] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    permutation<2> p2; p2.permute(0, 1);
    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator i =
        adapter.begin();
    const se2_t &el2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.is_symm() != (symm1 == symm2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "el2.get_perm() != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of a group in a 4-space onto a 2-space.
 **/
void so_merge_impl_perm_test::test_2n2nn_2(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_2n2nn_1(" << symm << ")";

    typedef se_perm<4, double> se4_t;
    typedef se_perm<2, double> se2_t;
    typedef so_merge<4, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
        so_merge_impl_t;

    try {

    se4_t el1(permutation<4>().permute(0, 1).permute(2, 3), symm);

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);

    mask<4> msk[2];
    msk[0][0] = true; msk[0][2] = true;
    msk[1][1] = true; msk[1][3] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    permutation<2> p2; p2.permute(0, 1);
    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator i =
        adapter.begin();
    const se2_t &el2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.is_symm() != symm) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "el2.get_perm() != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of a group in a 6-space onto 4-space.
 **/
void so_merge_impl_perm_test::test_nmk_1(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_nmk_1(" << symm << ")";

    typedef se_perm<6, double> se6_t;
    typedef se_perm<4, double> se4_t;
    typedef so_merge<6, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se6_t>
        so_merge_impl_t;

    try {

    se6_t el1a(permutation<6>().permute(0, 1), true);
    se6_t el1b(permutation<6>().permute(2, 3), true);
    se6_t el1c(
            permutation<6>().permute(0, 2).permute(1, 3).permute(4, 5), symm);

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);

    set1.insert(el1a);
    set1.insert(el1b);
    set1.insert(el1c);

    mask<6> msk[2];
    msk[0][0] = true; msk[0][1] = true;
    msk[1][2] = true; msk[1][3] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    permutation<4> p2; p2.permute(0, 1).permute(2, 3);
    symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
    symmetry_element_set_adapter<4, double, se4_t>::iterator i =
        adapter.begin();
    const se4_t &el2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.is_symm() != symm) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "el2.get_perm() != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Triple merge of a group in a 8-space onto 4-space.
 **/
void so_merge_impl_perm_test::test_nmk_2(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_perm_test::test_nmk_2(" << symm << ")";

    typedef se_perm<8, double> se8_t;
    typedef se_perm<4, double> se4_t;
    typedef so_merge<8, 7, 3, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se8_t>
        so_merge_impl_t;

    try {

    se8_t el1a(permutation<8>().permute(0, 2), symm);
    se8_t el1b(permutation<8>().permute(1, 3), symm);
    se8_t el1c(permutation<8>().permute(4, 5).permute(5, 7), true);
    se8_t el1d(permutation<8>().permute(4, 5)
            .permute(5, 6).permute(6, 7), true);
    symmetry_element_set<8, double> set1(se8_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);

    set1.insert(el1a);
    set1.insert(el1b);
    set1.insert(el1c);
    set1.insert(el1d);

    mask<8> msk[3];
    msk[0][0] = true; msk[0][1] = true;
    msk[1][2] = true; msk[1][3] = true;
    msk[2][4] = true; msk[2][5] = true; msk[2][7] = true;
    symmetry_operation_params<so_merge_t> params(set1, msk, set2);

    so_merge_impl_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    permutation<4> p2; p2.permute(0, 1);
    symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
    symmetry_element_set_adapter<4, double, se4_t>::iterator i =
        adapter.begin();
    const se4_t &el2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.is_symm() != true) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "el2.get_perm() != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
