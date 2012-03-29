#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/so_reduce_se_part.h>
#include "../compare_ref.h"
#include "so_reduce_se_part_test.h"


namespace libtensor {

void so_reduce_se_part_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2();
    test_nm1_1(true); test_nm1_1(false);
    test_nm1_2( true, true); test_nm1_2( true, false);
    test_nm1_2(false, true); test_nm1_2(false, false);
    test_nm1_3(true); test_nm1_3(false);
    test_nm1_4(true); test_nm1_4(false);
    test_nm1_5(true); test_nm1_5(false);
    test_nm1_6(true); test_nm1_6(false);
    test_nmk_1(true); test_nmk_1(false);
    test_nmk_2( true, true); test_nmk_2( true, false);
    test_nmk_2(false, true); test_nmk_2(false, false);

}


/**	\test Tests that a projection of an empty group yields an empty group
		of a lower order
 **/
void so_reduce_se_part_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_se_part_test::test_empty_1()";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        symmetry_element_set<4, double> set1(se4_t::k_sym_type);
        symmetry_element_set<2, double> set2(se2_t::k_sym_type);

        mask<4> msk; msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0);
        index<4> ia, ib; ib[2] = ib[3] = 2;
        index_range<4> ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(set1, msk,
                seq, ir, ir, set2);

        so_reduce_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Tests that a double projection of an empty group yields an empty group
		of a lower order
 **/
void so_reduce_se_part_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_se_part_test::test_empty_2()";

    typedef se_part<5, double> se5_t;
    typedef se_part<2, double> se2_t;
    typedef so_reduce<5, 3, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        symmetry_element_set<5, double> set1(se5_t::k_sym_type);
        symmetry_element_set<2, double> set2(se2_t::k_sym_type);

        mask<5> msk; msk[2] = msk[3] = msk[4] = true;
        sequence<5, size_t> seq(0); seq[3] = 1;
        index<5> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = ib[4] = 4;
        index_range<5> ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(set1, msk,
                seq, ir, ir, set2);

        so_reduce_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Projection of a 2-space in a single reduction step on a 1-space.
 **/
void so_reduce_se_part_test::test_nm1_1(bool sign)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_1(" << sign << ")";

    typedef se_part<2, double> se2_t;
    typedef se_part<1, double> se1_t;
    typedef so_reduce<2, 1, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se1_t> so_reduce_se_t;

    try {

        index<2> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5;
        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i1b)));
        mask<2> ma;
        ma[0] = true; ma[1] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<1> i2a, i2b;
        i2b[0] = 5;
        block_index_space<1> bisb(dimensions<1>(index_range<1>(i2a, i2b)));
        mask<1> mb; mb[0] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);


        se2_t ela(bisa, ma, 2);
        index<2> i00a, i01a, i02a, i03a;
        i02a[0] = 1; i01a[1] = 1;
        i03a[0] = 1; i03a[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i03a, sign ? tr0 : tr1);
        ela.mark_forbidden(i01a);
        ela.mark_forbidden(i02a);

        se1_t elb(bisb, mb, 2);
        index<1> i00b, i01b;
        i01b[0] = 1;
        elb.add_map(i00b, i01b, sign ? tr0 : tr1);


        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<1, double> setb(se1_t::k_sym_type);
        symmetry_element_set<1, double> setb_ref(se1_t::k_sym_type);
        seta.insert(ela);
        setb_ref.insert(elb);

        mask<2> m; m[1] = true;
        sequence<2, size_t> seq(0);
        index<2> bia, bib, ia, ib;
        bib[0] = bib[1] = 3;
        index_range<2> bir(bia, bib);
        index_range<2> ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, m,
                seq, bir, ir, setb);
        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<1>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/**	\test Projection of a 4-space in one reduction step onto a 2-space.
 **/
void so_reduce_se_part_test::test_nm1_2(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_2(" << s1 << ", " << s2 << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a, ma;
        m2a[0] = true; m2a[1] = true; m1a[2] = true; m1a[3] = true;
        ma[0]  = true; ma[1]  = true; ma[2]  = true; ma[3]  = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 9; i2b[1] = 9;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);
        bisb.split(mb, 7);
        bisb.split(mb, 8);

        se4_t ela(bisa, ma, 2);
        index<4> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a, i08a,
            i09a, i10a, i11a, i12a, i13a, i14a, i15a;
        i08a[0] = 1; i07a[1] = 1; i07a[2] = 1; i07a[3] = 1; // 1000
        i09a[0] = 1; i06a[1] = 1; i06a[2] = 1; i09a[3] = 1; // 1001
        i10a[0] = 1; i05a[1] = 1; i10a[2] = 1; i05a[3] = 1; // 1010
        i11a[0] = 1; i04a[1] = 1; i11a[2] = 1; i11a[3] = 1; // 1011
        i12a[0] = 1; i12a[1] = 1; i03a[2] = 1; i03a[3] = 1; // 1100
        i13a[0] = 1; i13a[1] = 1; i02a[2] = 1; i13a[3] = 1; // 1101
        i14a[0] = 1; i14a[1] = 1; i14a[2] = 1; i01a[3] = 1; // 1110
        i15a[0] = 1; i15a[1] = 1; i15a[2] = 1; i15a[3] = 1; // 1111
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i05a, s2 ? tr0 : tr1);
        ela.add_map(i05a, i10a, s1 == s2 ? tr0 : tr1);
        ela.add_map(i10a, i15a, s2 ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);
        ela.mark_forbidden(i03a); ela.mark_forbidden(i04a);
        ela.mark_forbidden(i06a); ela.mark_forbidden(i07a);
        ela.mark_forbidden(i08a); ela.mark_forbidden(i09a);
        ela.mark_forbidden(i11a); ela.mark_forbidden(i12a);
        ela.mark_forbidden(i13a); ela.mark_forbidden(i14a);

        se2_t elb(bisb, mb, 2);
        index<2> i00b, i01b, i02b, i03b;
        i02b[0] = 1; i01b[1] = 1;
        i03b[0] = 1; i03b[1] = 1;
        elb.add_map(i00b, i03b, s1 == s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        sequence<4, size_t> seq(0);
        index<4> bia, bib; bib[0] = bib[1] = 3; bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = ib[1] = 0; ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, m2a,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only not projected dims are partitioned)

 **/
void so_reduce_se_part_test::test_nm1_3(
        bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_3("")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a;
        m2a[0] = true; m2a[1] = true; m1a[2] = true; m1a[3] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);


        se4_t ela(bisa, m2a, 2);
        index<4> i00a, i04a, i08a, i10a;
        i08a[0] = 1; i04a[1] = 1; // 1000
        i10a[0] = 1; i10a[1] = 1; // 1100
        scalar_transf<double> tr0, tr1(-1.);

        ela.add_map(i00a, i10a, sign ? tr0 : tr1);
        ela.mark_forbidden(i04a); ela.mark_forbidden(i08a);

        se2_t elb(bisb, mb, 2);
        index<2> i00b, i01b, i02b, i03b;
        i02b[0] = 1; i01b[1] = 1;
        i03b[0] = 1; i03b[1] = 1;
        elb.add_map(i00b, i03b, sign ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0);
        index<4> bia, bib; bib[0] = bib[1] = 3; bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = ib[1] = 0; ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, msk,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only one partitioned dim, not projected)
 **/
void so_reduce_se_part_test::test_nm1_4(
        bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_4(" << sign << ")";

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 9; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a;
        m2a[0] = true; m1a[1] = true; m1a[2] = true; m1a[3] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 9;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> m1b, m2b;
        m2b[0] = true; m1b[1] = true;
        bisb.split(m1b, 2);
        bisb.split(m1b, 3);
        bisb.split(m1b, 5);
        bisb.split(m1b, 7);
        bisb.split(m1b, 8);
        bisb.split(m2b, 2);
        bisb.split(m2b, 3);
        bisb.split(m2b, 5);

        se4_t ela(bisa, m2a, 2);
        index<4> i00a, i08a;
        i08a[0] = 1;
        scalar_transf<double> tr0, tr1(-1.);

        ela.add_map(i00a, i08a, sign ? tr0 : tr1);

        se2_t elb(bisb, m2b, 2);
        index<2> i00b, i02b;
        i02b[0] = 1;
        elb.add_map(i00b, i02b, sign ? tr0 : tr1);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0);
        index<4> bia, bib; bib[0] = 3; bib[1] = bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = 0; ib[1] = ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, msk,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only one partitioned dim, projected)
 **/
void so_reduce_se_part_test::test_nm1_5(
        bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_5(" << sign << ")";

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a, m3a;
        m2a[0] = true; m2a[1] = true; m1a[2] = true; m1a[3] = true;
        m3a[1] = true; m3a[2] = true; m3a[3] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> m1b, m2b;
        m1b[0] = true; m1b[1] = true; m2b[1] = true;
        bisb.split(m1b, 2);
        bisb.split(m1b, 3);
        bisb.split(m1b, 5);

        se4_t ela(bisa, m3a, 2);
        index<4> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a;
        i04a[1] = 1; i03a[2] = 1; i03a[3] = 1;
        i05a[1] = 1; i02a[2] = 1; i05a[3] = 1;
        i06a[1] = 1; i06a[2] = 1; i01a[3] = 1;
        i07a[1] = 1; i07a[2] = 1; i07a[3] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i04a, sign ? tr0 : tr1);
        ela.add_map(i03a, i07a, sign ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);
        ela.mark_forbidden(i05a); ela.mark_forbidden(i06a);

        se2_t elb(bisb, m2b, 2);
        index<2> i00b, i01b;
        i01b[1] = 1;
        elb.add_map(i00b, i01b, sign ? tr0 : tr1);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        sequence<4, size_t> seq(0);
        index<4> bia, bib; bib[0] = bib[1] = 3; bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = ib[1] = 0; ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, m1a,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/**	\test Projection of a 4-space onto a 2-space in one step with partial
		partitioning (only projected dims are partitioned)
 **/
void so_reduce_se_part_test::test_nm1_6(
        bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nm1_6(" << sign << ")";

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a;
        m2a[0] = true; m2a[1] = true; m1a[2] = true; m1a[3] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> m1b;
        m1b[0] = true; m1b[1] = true;
        bisb.split(m1b, 2);
        bisb.split(m1b, 3);
        bisb.split(m1b, 5);

        se4_t ela(bisa, m1a, 2);
        index<4> i00a, i01a, i02a, i03a;
        i02a[2] = 1; i01a[3] = 1;
        i03a[2] = 1; i03a[3] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i03a, sign ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);

        seta.insert(ela);

        mask<4> msk; msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0);
        index<4> bia, bib; bib[0] = bib[1] = 3; bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = ib[1] = 0; ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, msk,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(! setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Projection of a 4-space onto a 2-space in two steps.
 **/
void so_reduce_se_part_test::test_nmk_1(
        bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nmk_1(" << sign << ")";

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 9; i1b[3] = 9;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> m1a, m2a, m3a;
        m2a[0] = true; m2a[1] = true; m1a[2] = true; m1a[3] = true;
        m3a[0] = true; m3a[1] = true; m3a[2] = true; m3a[3] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 9;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> m1b, m2b, m3b;
        m2b[0] = true; m1b[1] = true;
        m3b[0] = true; m3b[1] = true;
        bisb.split(m1b, 2);
        bisb.split(m1b, 3);
        bisb.split(m1b, 5);
        bisb.split(m1b, 7);
        bisb.split(m1b, 8);
        bisb.split(m2b, 2);
        bisb.split(m2b, 3);
        bisb.split(m2b, 5);


        se4_t ela(bisa, m3a, 2);
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
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i10a, sign ? tr0 : tr1);
        ela.add_map(i05a, i15a, sign ? tr0 : tr1);
        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);
        ela.mark_forbidden(i03a); ela.mark_forbidden(i04a);
        ela.mark_forbidden(i06a); ela.mark_forbidden(i07a);
        ela.mark_forbidden(i08a); ela.mark_forbidden(i09a);
        ela.mark_forbidden(i11a); ela.mark_forbidden(i12a);
        ela.mark_forbidden(i13a); ela.mark_forbidden(i14a);

        se2_t elb(bisb, m3b, 2);
        index<2> i00b, i01b, i02b, i03b;
        i02b[0] = 1; i01b[1] = 1;
        i03b[0] = 1; i03b[1] = 1;
        elb.add_map(i00b, i03b, sign ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[3] = msk[1] = true;
        sequence<4, size_t> seq(0); seq[1] = 1;
        index<4> bia, bib; bib[0] = bib[1] = 3; bib[2] = bib[3] = 5;
        index<4> ia, ib; ib[0] = ib[1] = 0; ib[2] = ib[3] = 1;
        index_range<4> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, msk,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Projection of a 6-space onto a 2-space in two steps.
 **/
void so_reduce_se_part_test::test_nmk_2(
        bool s1, bool s2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_part_test::test_nmk_2(" << s1 << ", " << s2 << ")";

    typedef se_part<6, double> se6_t;
    typedef se_part<2, double> se2_t;
    typedef so_reduce<6, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    try {

        index<6> i1a, i1b;
        i1b[0] = 5; i1b[1] = 9; i1b[2] = 9;
        i1b[3] = 5; i1b[4] = 9; i1b[5] = 9;
        block_index_space<6> bisa(dimensions<6>(index_range<6>(i1a, i1b)));
        mask<6> m1a, m2a, m3a;
        m2a[0] = true; m2a[3] = true;
        m1a[1] = true; m1a[2] = true; m1a[4] = true; m1a[5] = true;
        m3a[0] = true; m3a[1] = true; m3a[2] = true;
        m3a[3] = true; m3a[4] = true; m3a[5] = true;
        bisa.split(m1a, 2);
        bisa.split(m1a, 3);
        bisa.split(m1a, 5);
        bisa.split(m1a, 7);
        bisa.split(m1a, 8);
        bisa.split(m2a, 2);
        bisa.split(m2a, 3);
        bisa.split(m2a, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb;
        mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        scalar_transf<double> tr0, tr1(-1.);

        se6_t ela(bisa, m3a, 2);
        index<6> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a,
            i08a, i09a, i10a, i11a, i12a, i13a, i14a, i15a,
            i16a, i17a, i18a, i19a, i20a, i21a, i22a, i23a,
            i24a, i25a, i26a, i27a, i28a, i29a, i30a, i31a,
            i32a, i33a, i34a, i35a, i36a, i37a, i38a, i39a,
            i40a, i41a, i42a, i43a, i44a, i45a, i46a, i47a,
            i48a, i49a, i50a, i51a, i52a, i53a, i54a, i55a,
            i56a, i57a, i58a, i59a, i60a, i61a, i62a, i63a;

        i32a[0] = 1; i31a[1] = 1; i31a[2] = 1; // 100000
        i31a[3] = 1; i31a[4] = 1; i31a[5] = 1;
        i33a[0] = 1; i30a[1] = 1; i30a[2] = 1; // 100001
        i30a[3] = 1; i30a[4] = 1; i33a[5] = 1;
        i34a[0] = 1; i29a[1] = 1; i29a[2] = 1; // 100010
        i29a[3] = 1; i34a[4] = 1; i29a[5] = 1;
        i35a[0] = 1; i28a[1] = 1; i28a[2] = 1; // 100011
        i28a[3] = 1; i35a[4] = 1; i35a[5] = 1;
        i36a[0] = 1; i27a[1] = 1; i27a[2] = 1; // 100100
        i36a[3] = 1; i27a[4] = 1; i27a[5] = 1;
        i37a[0] = 1; i26a[1] = 1; i26a[2] = 1; // 100101
        i37a[3] = 1; i26a[4] = 1; i37a[5] = 1;
        i38a[0] = 1; i25a[1] = 1; i25a[2] = 1; // 100110
        i38a[3] = 1; i38a[4] = 1; i25a[5] = 1;
        i39a[0] = 1; i24a[1] = 1; i24a[2] = 1; // 100111
        i39a[3] = 1; i39a[4] = 1; i39a[5] = 1;
        i40a[0] = 1; i23a[1] = 1; i40a[2] = 1; // 101000
        i23a[3] = 1; i23a[4] = 1; i23a[5] = 1;
        i41a[0] = 1; i22a[1] = 1; i41a[2] = 1; // 101001
        i22a[3] = 1; i22a[4] = 1; i41a[5] = 1;
        i42a[0] = 1; i21a[1] = 1; i42a[2] = 1; // 101010
        i21a[3] = 1; i42a[4] = 1; i21a[5] = 1;
        i43a[0] = 1; i20a[1] = 1; i43a[2] = 1; // 101011
        i20a[3] = 1; i43a[4] = 1; i43a[5] = 1;
        i44a[0] = 1; i19a[1] = 1; i44a[2] = 1; // 101100
        i44a[3] = 1; i19a[4] = 1; i19a[5] = 1;
        i45a[0] = 1; i18a[1] = 1; i45a[2] = 1; // 101101
        i45a[3] = 1; i18a[4] = 1; i45a[5] = 1;
        i46a[0] = 1; i17a[1] = 1; i46a[2] = 1; // 101110
        i46a[3] = 1; i46a[4] = 1; i17a[5] = 1;
        i47a[0] = 1; i16a[1] = 1; i47a[2] = 1; // 101111
        i47a[3] = 1; i47a[4] = 1; i47a[5] = 1;
        i48a[0] = 1; i48a[1] = 1; i15a[2] = 1; // 110000
        i15a[3] = 1; i15a[4] = 1; i15a[5] = 1;
        i49a[0] = 1; i49a[1] = 1; i14a[2] = 1; // 110001
        i14a[3] = 1; i14a[4] = 1; i49a[5] = 1;
        i50a[0] = 1; i50a[1] = 1; i13a[2] = 1; // 110010
        i13a[3] = 1; i50a[4] = 1; i13a[5] = 1;
        i51a[0] = 1; i51a[1] = 1; i12a[2] = 1; // 110011
        i12a[3] = 1; i51a[4] = 1; i51a[5] = 1;
        i52a[0] = 1; i52a[1] = 1; i11a[2] = 1; // 110100
        i52a[3] = 1; i11a[4] = 1; i11a[5] = 1;
        i53a[0] = 1; i53a[1] = 1; i10a[2] = 1; // 110101
        i53a[3] = 1; i10a[4] = 1; i53a[5] = 1;
        i54a[0] = 1; i54a[1] = 1; i09a[2] = 1; // 110110
        i54a[3] = 1; i54a[4] = 1; i09a[5] = 1;
        i55a[0] = 1; i55a[1] = 1; i08a[2] = 1; // 110111
        i55a[3] = 1; i55a[4] = 1; i55a[5] = 1;
        i56a[0] = 1; i56a[1] = 1; i56a[2] = 1; // 111000
        i07a[3] = 1; i07a[4] = 1; i07a[5] = 1;
        i57a[0] = 1; i57a[1] = 1; i57a[2] = 1; // 111001
        i06a[3] = 1; i06a[4] = 1; i57a[5] = 1;
        i58a[0] = 1; i58a[1] = 1; i58a[2] = 1; // 111010
        i05a[3] = 1; i58a[4] = 1; i05a[5] = 1;
        i59a[0] = 1; i59a[1] = 1; i59a[2] = 1; // 111011
        i04a[3] = 1; i59a[4] = 1; i59a[5] = 1;
        i60a[0] = 1; i60a[1] = 1; i60a[2] = 1; // 111100
        i60a[3] = 1; i03a[4] = 1; i03a[5] = 1;
        i61a[0] = 1; i61a[1] = 1; i61a[2] = 1; // 111101
        i61a[3] = 1; i02a[4] = 1; i61a[5] = 1;
        i62a[0] = 1; i62a[1] = 1; i62a[2] = 1; // 111110
        i62a[3] = 1; i62a[4] = 1; i01a[5] = 1;
        i63a[0] = 1; i63a[1] = 1; i63a[2] = 1; // 111111
        i63a[3] = 1; i63a[4] = 1; i63a[5] = 1;

        ela.add_map(i00a, i07a, s2 ? tr0 : tr1);
        ela.add_map(i07a, i56a, s1 == s2 ? tr0 : tr1);
        ela.add_map(i56a, i63a, s2 ? tr0 : tr1);

        ela.mark_forbidden(i01a); ela.mark_forbidden(i02a);
        ela.mark_forbidden(i03a); ela.mark_forbidden(i04a);
        ela.mark_forbidden(i05a); ela.mark_forbidden(i06a);
        ela.mark_forbidden(i08a); ela.mark_forbidden(i09a);
        ela.mark_forbidden(i10a); ela.mark_forbidden(i11a);
        ela.mark_forbidden(i12a); ela.mark_forbidden(i13a);
        ela.mark_forbidden(i14a); ela.mark_forbidden(i15a);
        ela.mark_forbidden(i16a); ela.mark_forbidden(i17a);
        ela.mark_forbidden(i18a); ela.mark_forbidden(i19a);
        ela.mark_forbidden(i20a); ela.mark_forbidden(i21a);
        ela.mark_forbidden(i22a); ela.mark_forbidden(i23a);
        ela.mark_forbidden(i24a); ela.mark_forbidden(i25a);
        ela.mark_forbidden(i26a); ela.mark_forbidden(i27a);
        ela.mark_forbidden(i28a); ela.mark_forbidden(i29a);
        ela.mark_forbidden(i30a); ela.mark_forbidden(i31a);
        ela.mark_forbidden(i32a); ela.mark_forbidden(i33a);
        ela.mark_forbidden(i34a); ela.mark_forbidden(i35a);
        ela.mark_forbidden(i36a); ela.mark_forbidden(i37a);
        ela.mark_forbidden(i38a); ela.mark_forbidden(i39a);
        ela.mark_forbidden(i40a); ela.mark_forbidden(i41a);
        ela.mark_forbidden(i42a); ela.mark_forbidden(i43a);
        ela.mark_forbidden(i44a); ela.mark_forbidden(i45a);
        ela.mark_forbidden(i46a); ela.mark_forbidden(i47a);
        ela.mark_forbidden(i48a); ela.mark_forbidden(i49a);
        ela.mark_forbidden(i50a); ela.mark_forbidden(i51a);
        ela.mark_forbidden(i52a); ela.mark_forbidden(i53a);
        ela.mark_forbidden(i54a); ela.mark_forbidden(i55a);
        ela.mark_forbidden(i57a); ela.mark_forbidden(i58a);
        ela.mark_forbidden(i59a); ela.mark_forbidden(i60a);
        ela.mark_forbidden(i61a); ela.mark_forbidden(i62a);

        se2_t elb(bisb, mb, 2);
        index<2> i00b, i01b, i02b, i03b;
        i02b[0] = 1; i01b[1] = 1;
        i03b[0] = 1; i03b[1] = 1;
        elb.add_map(i00b, i03b, s1 == s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b); elb.mark_forbidden(i02b);

        symmetry_element_set<6, double> seta(se6_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<6> msk; msk[1] = true; msk[4] = msk[2] = msk[5] = true;
        sequence<6, size_t> seq(0); seq[2] = seq[5] = 1;
        index<6> bia, bib, ia, ib;
        bib[0] = bib[3] = 3; bib[1] = bib[2] = bib[4] = bib[5] = 5;
        ib[0] = ib[3] = 0; ib[1] = ib[2] = ib[4] = ib[5] = 1;
        index_range<6> bir(bia, bib), ir(ia, ib);
        symmetry_operation_params<so_reduce_t> params(seta, msk,
                seq, bir, ir, setb);

        so_reduce_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
