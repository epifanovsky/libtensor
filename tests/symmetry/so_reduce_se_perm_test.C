#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_reduce_se_perm.h>
#include "../compare_ref.h"
#include "so_reduce_se_perm_test.h"


namespace libtensor {


void so_reduce_se_perm_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2();
    test_nm1_1(true); test_nm1_1(false);
    test_nm1_2(true); test_nm1_2(false);
    test_nmk_1(true); test_nmk_1(false);
    test_nmk_2(true); test_nmk_2(false);
    test_nmk_3( true, true); test_nmk_3( true, false);
    test_nmk_3(false, true); test_nmk_3(false, false);
}


/** \test Tests that a single reduction step on an empty group yields an empty
        group of lower order
 **/
void so_reduce_se_perm_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_se_perm_test::test_empty_1()";

    typedef se_perm<4, double> se4_t;
    typedef se_perm<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    mask<4> msk; msk[0] = msk[1] = true;
    sequence<4, size_t> seq(0);
    index<4> i1a, i1b; i1b[0] = i1b[1] = i1b[2] = i1b[3] = 3;
    index_range<4> ir(i1a, i1b);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected an empty set.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Tests that two reduction steps on an empty group yield an empty
        group of lower order
 **/
void so_reduce_se_perm_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_se_perm_test::test_empty_2()";

    typedef se_perm<6, double> se6_t;
    typedef se_perm<2, double> se2_t;
    typedef so_reduce<6, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    mask<6> msk; msk[0] = msk[2] = msk[1] = msk[3] = true;
    sequence<6, size_t> seq(0); seq[1] = seq[3] = 1;
    index<6> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = ib[4] = ib[5] = 4;
    index_range<6> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected an empty set.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Projection of one 2-cycle in a 2-space onto a 1-space. Expected
        result: C1 in 1-space. Symmetric elements.
 **/
void so_reduce_se_perm_test::test_nm1_1(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_perm_test::test_nm1_1(" << symm << ")";

    typedef se_perm<1, double> se1_t;
    typedef se_perm<2, double> se2_t;
    typedef so_reduce<2, 1, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se1_t> so_reduce_se_t;

    try {

    permutation<2> p1; p1.permute(0, 1);
    scalar_transf<double> tr(symm ? 1.0 : -1.0);
    se2_t elem1(p1, tr);

    symmetry_element_set<2, double> set1(se2_t::k_sym_type);
    symmetry_element_set<1, double> set2(se1_t::k_sym_type);

    set1.insert(elem1);

    mask<2> msk; msk[0] = true;
    sequence<2, size_t> seq(0);
    index<2> ia, ib; ib[0] = ib[1] = 2;
    index_range<2> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected an empty set.");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Projection of S2(+/-) x S2(+/-) in 4-space onto a 2-space in a single
        step. Expected result: C1 in 1-space. Symmetric elements.
 **/
void so_reduce_se_perm_test::test_nm1_2(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_perm_test::test_nm1_2(" << symm << ")";

    typedef se_perm<2, double> se2_t;
    typedef se_perm<4, double> se4_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

    scalar_transf<double> tr(symm ? 1.0 : -1.0);
    se4_t el1(permutation<4>().permute(0, 1).permute(2, 3), tr);

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);

    mask<4> msk; msk[2] = msk[3] = true;
    sequence<4, size_t> seq(0);
    index<4> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = 2;
    index_range<4> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator i =
        adapter.begin();
    const se2_t &el2 = adapter.get_elem(i);
    permutation<2> p2; p2.permute(0, 1);
    i++;
    if(i != adapter.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected only one element.");
    }
    if(el2.get_transf() != tr) {
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


/** \test Projection of a 2-cycle in a 5-space in two reduction steps onto a
        2-space untouched by the masks. Expected result: 2-cycle in 2-space.
 **/
void so_reduce_se_perm_test::test_nmk_1(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_perm_test::test_nmk_1(" << symm << ")";

    typedef se_perm<2, double> se2_t;
    typedef se_perm<5, double> se5_t;
    typedef so_reduce<5, 3, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t>
        so_reduce_se_t;

    try {

    permutation<5> p1; p1.permute(0, 1);
    scalar_transf<double> tr(symm ? 1.0 : -1.0);
    se5_t el1(p1, tr);

    symmetry_element_set<5, double> set1(se5_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);

    mask<5> msk; msk[2] = msk[3] = msk[4] = true;
    sequence<5, size_t> seq(0); seq[4] = 1;
    index<5> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = ib[4] = 4;
    index_range<5> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

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
    if(el2.get_transf() != tr) {
        fail_test(tnss.str().c_str(),
                __FILE__, __LINE__, "el2.get_transf() != tr");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "el2.perm != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Projection of a group in a 6-space in 3 reduction steps onto a
        3-space with one dimension out. Expected result: S2(+/-) in 3-space.
 **/
void so_reduce_se_perm_test::test_nmk_2(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_perm_test::test_nmk_2(" << symm << ")";

    typedef se_perm<6, double> se6_t;
    typedef se_perm<3, double> se3_t;
    typedef so_reduce<6, 3, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se3_t>
        so_reduce_se_t;

    try {

    scalar_transf<double> tr0, tr1(-1.);
    se6_t el1a(permutation<6>().permute(0, 1).permute(1, 3), tr0);
    se6_t el1b(permutation<6>().permute(0, 1), symm ? tr0 : tr1);

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<3, double> set2(se3_t::k_sym_type);

    set1.insert(el1a);
    set1.insert(el1b);

    mask<6> msk; msk[3] = msk[4] = msk[5] = true;
    sequence<6, size_t> seq; seq[3] = 1; seq[4] = 2;
    index<6> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = ib[4] = ib[5] = 4;
    index_range<6> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

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
    if(el2.get_transf() != (symm ? tr0 : tr1)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Wrong sign.");
    }
    if(!el2.get_perm().equals(p2)) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, "el2.perm != p2");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Projection of a group in 8-space in two reduction steps on to 4-space.
 **/
void so_reduce_se_perm_test::test_nmk_3(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_perm_test::test_nmk_3(" <<
            symm1 << ", " << symm2 << ")";

    typedef se_perm<8, double> se8_t;
    typedef se_perm<4, double> se4_t;
    typedef so_reduce<8, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se4_t>
        so_reduce_se_t;

    try {

    scalar_transf<double> tr1(symm1 ? 1.0 : -1.0), tr2(symm2 ? 1.0 : -1.0);
    se8_t el1a(permutation<8>().permute(0, 1).permute(2, 3), tr1);
    se8_t el1b(permutation<8>().permute(4, 5).permute(6, 7), tr2);

    permutation<4> p2;
    p2.permute(0, 1).permute(2, 3);
    scalar_transf<double> trx(tr1); trx.transform(tr2);
    se4_t el2(permutation<4>().permute(0, 1).permute(2, 3), trx);

    symmetry_element_set<8, double> set1(se8_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

    set1.insert(el1a);
    set1.insert(el1b);
    set2_ref.insert(el2);

    mask<8> msk; msk[2] = msk[3] = msk[6] = msk[7] = true;
    sequence<8, size_t> seq(0); seq[3] = 1; seq[7] = 1;
    index<8> i1, i2;
    i2[0] = i2[1] = i2[2] = i2[3] = i2[4] = i2[5] = i2[6] = i2[7] = 3;
    index_range<8> ir(i1, i2);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "Expected a non-empty set.");
    }

    index<4> i1a, i1b; i1b[0] = i1b[1] = i1b[2] = i1b[3] = 3;
    block_index_space<4> bis(dimensions<4>(index_range<4>(i1a, i1b)));
    mask<4> m;
    m[0] = m[1] = m[2] = m[3] = true;
    bis.split(m, 1);
    bis.split(m, 2);
    bis.split(m, 3);

    compare_ref<4>::compare(tnss.str().c_str(), bis, set2, set2_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

