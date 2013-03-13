#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_symmetrize_se_part.h>
#include "../compare_ref.h"
#include "so_symmetrize_se_part_test.h"


namespace libtensor {

void so_symmetrize_se_part_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2();
    test_sym2_1(true); test_sym2_1(false);
    test_sym2_2();
}


/** \test Tests that pair symmetrization of an empty group yields an
        empty group
 **/
void so_symmetrize_se_part_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_symmetrize_se_part_test::test_empty_1()";

    typedef se_part<5, double> se5_t;
    typedef so_symmetrize<5, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se5_t> so_symmetrize_se_t;

    try {

        symmetry_element_set<5, double> set1(se5_t::k_sym_type);
        symmetry_element_set<5, double> set2(se5_t::k_sym_type);


        sequence<5, size_t> idxgrp(0), symidx(0);
        idxgrp[0] = idxgrp[1] = 1; idxgrp[3] = idxgrp[4] = 2;
        symidx[0] = symidx[3] = 1; symidx[1] = symidx[4] = 2;
        scalar_transf<double> trp(1.0), trc(1.0);

        symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
                symidx, trp, trc, set2);

        so_symmetrize_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Tests that a symmetrization of an empty group yields an empty group
 **/
void so_symmetrize_se_part_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_symmetrize_se_part_test::test_empty_2()";

    typedef se_part<5, double> se5_t;
    typedef so_symmetrize<5, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se5_t> so_symmetrize_se_t;

    try {

        symmetry_element_set<5, double> set1(se5_t::k_sym_type);
        symmetry_element_set<5, double> set2(se5_t::k_sym_type);

        sequence<5, size_t> idxgrp(0), symidx(0);
        idxgrp[0] = 1; idxgrp[1] = 2; idxgrp[3] = 3;
        symidx[0] = symidx[1] = symidx[3] = 1;
        scalar_transf<double> tr;
        symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
                symidx, tr, tr, set2);

        so_symmetrize_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Pair symmetrization of a group in 4-space.
 **/
void so_symmetrize_se_part_test::test_sym2_1(bool sign)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_part_test::test_sym2_1(" << sign << ")";

    typedef se_part<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma;
        ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<4> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a, i08a,
            i09a, i10a, i11a, i12a, i13a, i14a, i15a;
        i08a[0] = 1; i07a[1] = 1; i07a[2] = 1; i07a[3] = 1;
        i09a[0] = 1; i06a[1] = 1; i06a[2] = 1; i09a[3] = 1;
        i10a[0] = 1; i05a[1] = 1; i10a[2] = 1; i05a[3] = 1;
        i11a[0] = 1; i04a[1] = 1; i11a[2] = 1; i11a[3] = 1;
        i12a[0] = 1; i12a[1] = 1; i03a[2] = 1; i03a[3] = 1;
        i13a[0] = 1; i13a[1] = 1; i02a[2] = 1; i13a[3] = 1;
        i14a[0] = 1; i14a[1] = 1; i14a[2] = 1; i01a[3] = 1;
        i15a[0] = 1; i15a[1] = 1; i15a[2] = 1; i15a[3] = 1;

        scalar_transf<double> tr0, tr1(-1.0);

        se4_t ela(bisa, ma, 2);
        ela.add_map(i00a, i10a, sign ? tr0 : tr1);
        ela.add_map(i05a, i15a, sign ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);
        ela.mark_forbidden(i03a); ela.mark_forbidden(i04a);
        ela.mark_forbidden(i06a); ela.mark_forbidden(i07a);
        ela.mark_forbidden(i08a); ela.mark_forbidden(i09a);
        ela.mark_forbidden(i11a); ela.mark_forbidden(i12a);
        ela.mark_forbidden(i13a); ela.mark_forbidden(i14a);

        se4_t elb(bisa, ma, 2);
        elb.add_map(i00a, i10a, sign ? tr0 : tr1);
        elb.add_map(i05a, i15a, sign ? tr0 : tr1);
        elb.mark_forbidden(i01a); elb.mark_forbidden(i02a);
        elb.mark_forbidden(i03a); elb.mark_forbidden(i04a);
        elb.mark_forbidden(i06a); elb.mark_forbidden(i07a);
        elb.mark_forbidden(i08a); elb.mark_forbidden(i09a);
        elb.mark_forbidden(i11a); elb.mark_forbidden(i12a);
        elb.mark_forbidden(i13a); elb.mark_forbidden(i14a);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<4, double> setb(se4_t::k_sym_type);
        symmetry_element_set<4, double> setb_ref(se4_t::k_sym_type);
        seta.insert(ela);
        setb_ref.insert(elb);

        sequence<4, size_t> idxgrp(0), symidx(0);
        idxgrp[0] = idxgrp[1] = 1; idxgrp[2] = idxgrp[3] = 2;
        symidx[0] = symidx[2] = 1; symidx[1] = symidx[3] = 2;

        symmetry_operation_params<so_symmetrize_t> params(seta, idxgrp,
                symidx, tr1, tr1, setb);
        so_symmetrize_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<4>::compare(tnss.str().c_str(), bisa, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Anti-symmetrization of a group in 4-space.
 **/
void so_symmetrize_se_part_test::test_sym2_2()
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_part_test::test_sym2_2()";

    typedef se_part<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 9; i1b[1] = 9; i1b[2] = 19; i1b[3] = 19;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma, mb;
        ma[0] = true; ma[1] = true; mb[2] = true; mb[3] = true;
        bisa.split(ma, 3);
        bisa.split(ma, 5);
        bisa.split(ma, 8);
        bisa.split(mb, 6);
        bisa.split(mb, 10);
        bisa.split(mb, 16);

        index<4> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a, i08a,
            i09a, i10a, i11a, i12a, i13a, i14a, i15a;
        i08a[0] = 1; i07a[1] = 1; i07a[2] = 1; i07a[3] = 1;
        i09a[0] = 1; i06a[1] = 1; i06a[2] = 1; i09a[3] = 1;
        i10a[0] = 1; i05a[1] = 1; i10a[2] = 1; i05a[3] = 1;
        i11a[0] = 1; i04a[1] = 1; i11a[2] = 1; i11a[3] = 1;
        i12a[0] = 1; i12a[1] = 1; i03a[2] = 1; i03a[3] = 1;
        i13a[0] = 1; i13a[1] = 1; i02a[2] = 1; i13a[3] = 1;
        i14a[0] = 1; i14a[1] = 1; i14a[2] = 1; i01a[3] = 1;
        i15a[0] = 1; i15a[1] = 1; i15a[2] = 1; i15a[3] = 1;

        scalar_transf<double> tr0, tr1(-1.0);

        mask<4> m1010, m0101, m1111;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        se4_t ela(bisa, m1010, 2);
        ela.add_map(i00a, i10a, tr0);
        ela.mark_forbidden(i02a);
        ela.mark_forbidden(i08a);

        se4_t elb(bisa, m0101, 2);
        elb.add_map(i00a, i05a, tr0);
        elb.mark_forbidden(i01a);
        elb.mark_forbidden(i04a);

        se4_t elc(bisa, m1111, 2);
        elc.add_map(i00a, i15a, tr1);
        elc.add_map(i05a, i10a, tr0);
        elc.add_map(i06a, i09a, tr0);
        elc.mark_forbidden(i01a);
        elc.mark_forbidden(i02a);
        elc.mark_forbidden(i03a);
        elc.mark_forbidden(i04a);
        elc.mark_forbidden(i07a);
        elc.mark_forbidden(i08a);
        elc.mark_forbidden(i11a);
        elc.mark_forbidden(i12a);
        elc.mark_forbidden(i13a);
        elc.mark_forbidden(i14a);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<4, double> setc(se4_t::k_sym_type);
        symmetry_element_set<4, double> setc_ref(se4_t::k_sym_type);
        seta.insert(ela);
        seta.insert(elb);
        setc_ref.insert(elc);

        sequence<4, size_t> idxgrp(0), symidx(0);
        idxgrp[0] = 1; idxgrp[1] = 2;
        symidx[0] = symidx[1] = 1;

        symmetry_operation_params<so_symmetrize_t> params(seta, idxgrp,
                symidx, tr0, tr1, setc);
        so_symmetrize_se_t().perform(params);

        if(setc.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<4>::compare(tnss.str().c_str(), bisa, setc, setc_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
