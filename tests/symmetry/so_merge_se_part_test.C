#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_merge_se_part.h>
#include "../compare_ref.h"
#include "so_merge_se_part_test.h"


namespace libtensor {

void so_merge_se_part_test::perform() {

    test_empty_1();
    test_empty_2();
    test_nm1_1(true); test_nm1_1(false);
    test_nm1_2(true); test_nm1_2(false);
    test_2n2nn_1(true,  true); test_2n2nn_1(false,  true);
    test_2n2nn_1(true, false); test_2n2nn_1(false, false);
    test_2n2nn_2(true,  true); test_2n2nn_2(false,  true);
    test_2n2nn_2(true, false); test_2n2nn_2(false, false);
    test_2n2nn_3(true); test_2n2nn_3(false);
    test_nmk_1(true); test_nmk_1(false);
    test_nmk_2(true,  true); test_nmk_2(false,  true);
    test_nmk_2(true, false); test_nmk_2(false, false);
}


/** \test Tests that a single merge of 2 dim of an empty partition set yields
        an empty partition set of lower order
 **/
void so_merge_se_part_test::test_empty_1() {

    static const char *testname = "so_merge_se_part_test::test_empty_1()";

    typedef se_part<4, double> se4_t;
    typedef se_part<3, double> se3_t;
    typedef so_merge<4, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t>
    so_merge_se_t;

    try {

        symmetry_element_set<4, double> set1(se4_t::k_sym_type);
        symmetry_element_set<3, double> set2(se3_t::k_sym_type);

        mask<4> msk; msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0);
        symmetry_operation_params<so_merge_t> params(set1, msk, seq, set2);

        so_merge_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Tests that a double merge of dimensions of an empty partition set
        yields an empty partition set of lower order
 **/
void so_merge_se_part_test::test_empty_2() {

    static const char *testname = "so_merge_se_part_test::test_empty_2()";

    typedef se_part<5, double> se5_t;
    typedef se_part<3, double> se3_t;
    typedef so_merge<5, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t>
    so_merge_se_t;

    try {

        symmetry_element_set<5, double> set1(se5_t::k_sym_type);
        symmetry_element_set<3, double> set2(se3_t::k_sym_type);

        mask<5> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
        sequence<5, size_t> seq(0); seq[1] = seq[3] = 1;
        symmetry_operation_params<so_merge_t> params(set1, msk, seq, set2);

        so_merge_se_t().perform(params);

        if(!set2.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Single merge of 2 dim of a 3-space on a 2-space.
 **/
void so_merge_se_part_test::test_nm1_1(bool sign) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_nm1_1(" << sign << ")";

    typedef se_part<3, double> se3_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<3, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se2_t>
    so_merge_se_t;

    try {

        libtensor::index<3> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5;
        block_index_space<3> bisa(dimensions<3>(index_range<3>(i1a, i1b)));
        mask<3> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb;
        mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se3_t ela(bisa, ma, 2);
        libtensor::index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i000, i001, tr0);
        ela.add_map(i001, i110, sign ? tr0 : tr1);
        ela.add_map(i110, i111, tr0);
        ela.mark_forbidden(i010);
        ela.mark_forbidden(i011);
        ela.mark_forbidden(i100);
        ela.mark_forbidden(i101);

        se2_t elb(bisb, mb, 2);
        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        elb.add_map(i00, i11, sign ? tr0 : tr1);
        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<3> mc; mc[1] = mc[2] = true;
        sequence<3, size_t> seq(0);
        symmetry_operation_params<so_merge_t> params(seta, mc, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Single merge of 3 dim of a 3-space on a 1-space.
 **/
void so_merge_se_part_test::test_nm1_2(bool sign) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_nm1_1(" << tnss << ")";

    typedef se_part<3, double> se3_t;
    typedef se_part<1, double> se1_t;
    typedef so_merge<3, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se1_t> so_merge_se_t;

    try {

        libtensor::index<3> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5;
        block_index_space<3> bisa(dimensions<3>(index_range<3>(i1a, i1b)));
        mask<3> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<1> i2a, i2b;
        i2b[0] = 5;
        block_index_space<1> bisb(dimensions<1>(index_range<1>(i2a, i2b)));
        mask<1> mb;
        mb[0] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se3_t ela(bisa, ma, 2);
        libtensor::index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i000, i001, tr0);
        ela.add_map(i001, i110, sign ? tr0 : tr1);
        ela.add_map(i110, i111, tr0);
        ela.mark_forbidden(i010);
        ela.mark_forbidden(i011);
        ela.mark_forbidden(i100);
        ela.mark_forbidden(i101);

        se1_t elb(bisb, mb, 2);
        libtensor::index<1> i0, i1;
        i1[0] = 1;
        elb.add_map(i0, i1, sign ? tr0 : tr1);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<1, double> setb(se1_t::k_sym_type);
        symmetry_element_set<1, double> setb_ref(se1_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<3> msk; msk[0] = msk[1] = msk[2] = true;
        sequence<3, size_t> seq(0);
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<1>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of 4 dim of a 4-space on a 2-space.
 **/
void so_merge_se_part_test::test_2n2nn_1(bool s1, bool s2) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_2n2nn_1(" << s1 << ", " << s2 << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se2_t> so_merge_se_t;

    try {

        libtensor::index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma; ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se4_t ela(bisa, ma, 2);
        libtensor::index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i0000, i0011, s2 ? tr0 : tr1);
        ela.add_map(i0011, i1100, s1 == s2 ? tr0 : tr1);
        ela.add_map(i1100, i1111, s2 ? tr0 : tr1);
        ela.mark_forbidden(i0001);
        ela.mark_forbidden(i0010);
        ela.mark_forbidden(i0100);
        ela.mark_forbidden(i0101);
        ela.mark_forbidden(i0110);
        ela.mark_forbidden(i0111);
        ela.mark_forbidden(i1000);
        ela.mark_forbidden(i1001);
        ela.mark_forbidden(i1010);
        ela.mark_forbidden(i1011);
        ela.mark_forbidden(i1101);
        ela.mark_forbidden(i1110);

        se2_t elb(bisb, mb, 2);
        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        elb.add_map(i00, i11, s1 == s2 ? tr0 : tr1);
        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0); seq[1] = seq[3] = 1;
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of 4 dim of a 4-space on a 2-space.
 **/
void so_merge_se_part_test::test_2n2nn_2(bool s1, bool s2) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_2n2nn_2(" << s1 << ", " << s2 << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se2_t>
    so_merge_se_t;

    try {

        libtensor::index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma; ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se4_t ela(bisa, ma, 2);
        libtensor::index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        if (s1) {
            ela.add_map(i0000, i1100, tr0);
            ela.add_map(i0001, i1101, tr0);
            ela.add_map(i0010, i1110, tr0);
            ela.add_map(i0011, i1111, tr0);
        }
        if (s2) {
            ela.add_map(i0000, i0011, tr0);
            ela.add_map(i0100, i0111, tr0);
            ela.add_map(i1000, i1011, tr0);
            ela.add_map(i1100, i1111, tr0);
        }
        if (s1 == s2 && ! s1) {
            ela.add_map(i0000, i1111, s1 ? tr0 : tr1);
            ela.add_map(i0011, i1100, s1 ? tr0 : tr1);
        }
        ela.mark_forbidden(i0101);
        ela.mark_forbidden(i0110);
        ela.mark_forbidden(i1001);
        ela.mark_forbidden(i1010);

        se2_t elb(bisb, mb, 2);
        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        if (s1 == s2)
            elb.add_map(i00, i11, s1 ? tr0 : tr1);

        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0); seq[1] = seq[3] = 1;
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Double merge of 4 dim of a 4-space on a 2-space.
 **/
void so_merge_se_part_test::test_2n2nn_3(bool sign) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_2n2nn_3(" << sign << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se2_t> so_merge_se_t;

    try {

        libtensor::index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma; ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        mask<4> m; m[0] = m[1] = true;
        se4_t ela(bisa, m, 2);
        libtensor::index<4> i0000, i0100, i1000, i1100;
        i1000[0] = 1; i0100[1] = 1;
        i1100[0] = 1; i1100[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i0000, i1100, sign ? tr0 : tr1);
        ela.mark_forbidden(i0100);
        ela.mark_forbidden(i1000);

        se2_t elb(bisb, mb, 2);
        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
        sequence<4, size_t> seq(0); seq[1] = seq[3] = 1;
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<2>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Double merge of 4 dim of a 5-space on a 3-space.
 **/
void so_merge_se_part_test::test_nmk_1(bool sign) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_nmk_1(" << sign << ")";

    typedef se_part<5, double> se5_t;
    typedef se_part<3, double> se3_t;
    typedef so_merge<5, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t> so_merge_se_t;

    try {

        libtensor::index<5> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5; i1b[4] = 5;
        block_index_space<5> bisa(dimensions<5>(index_range<5>(i1a, i1b)));
        mask<5> ma;
        ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true; ma[4] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<3> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5; i2b[2] = 5;
        block_index_space<3> bisb(dimensions<3>(index_range<3>(i2a, i2b)));
        mask<3> mb; mb[0] = true; mb[1] = true; mb[2] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se5_t ela(bisa, ma, 2);
        libtensor::index<5> i00a, i01a, i02a, i03a, i04a, i05a, i06a, i07a,
            i08a, i09a, i10a, i11a, i12a, i13a, i14a, i15a,
            i16a, i17a, i18a, i19a, i20a, i21a, i22a, i23a,
            i24a, i25a, i26a, i27a, i28a, i29a, i30a, i31a;
        i16a[0] = 1; i15a[1] = 1; i15a[2] = 1; i15a[3] = 1; i15a[4] = 1; // 10000
        i17a[0] = 1; i14a[1] = 1; i14a[2] = 1; i14a[3] = 1; i17a[4] = 1; // 10001
        i18a[0] = 1; i13a[1] = 1; i13a[2] = 1; i18a[3] = 1; i13a[4] = 1; // 10010
        i19a[0] = 1; i12a[1] = 1; i12a[2] = 1; i19a[3] = 1; i19a[4] = 1; // 10011
        i20a[0] = 1; i11a[1] = 1; i20a[2] = 1; i11a[3] = 1; i11a[4] = 1; // 10100
        i21a[0] = 1; i10a[1] = 1; i21a[2] = 1; i10a[3] = 1; i21a[4] = 1; // 10101
        i22a[0] = 1; i09a[1] = 1; i22a[2] = 1; i22a[3] = 1; i09a[4] = 1; // 10110
        i23a[0] = 1; i08a[1] = 1; i23a[2] = 1; i23a[3] = 1; i23a[4] = 1; // 10111
        i24a[0] = 1; i24a[1] = 1; i07a[2] = 1; i07a[3] = 1; i07a[4] = 1; // 11000
        i25a[0] = 1; i25a[1] = 1; i06a[2] = 1; i06a[3] = 1; i25a[4] = 1; // 11001
        i26a[0] = 1; i26a[1] = 1; i05a[2] = 1; i26a[3] = 1; i05a[4] = 1; // 11010
        i27a[0] = 1; i27a[1] = 1; i04a[2] = 1; i27a[3] = 1; i27a[4] = 1; // 11011
        i28a[0] = 1; i28a[1] = 1; i28a[2] = 1; i03a[3] = 1; i03a[4] = 1; // 11100
        i29a[0] = 1; i29a[1] = 1; i29a[2] = 1; i02a[3] = 1; i29a[4] = 1; // 11101
        i30a[0] = 1; i30a[1] = 1; i30a[2] = 1; i30a[3] = 1; i01a[4] = 1; // 11110
        i31a[0] = 1; i31a[1] = 1; i31a[2] = 1; i31a[3] = 1; i31a[4] = 1; // 11111
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(i00a, i01a, sign ? tr0 : tr1);
        ela.add_map(i01a, i10a, sign ? tr0 : tr1);
        ela.add_map(i02a, i03a, sign ? tr0 : tr1);
        ela.add_map(i03a, i08a, tr0);
        ela.add_map(i04a, i05a, sign ? tr0 : tr1);
        ela.add_map(i05a, i16a, tr0);
        ela.add_map(i08a, i09a, sign ? tr0 : tr1);
        ela.add_map(i10a, i11a, sign ? tr0 : tr1);
        ela.add_map(i11a, i20a, sign ? tr0 : tr1);
        ela.add_map(i14a, i15a, sign ? tr0 : tr1);
        ela.add_map(i15a, i26a, tr0);
        ela.add_map(i16a, i17a, sign ? tr0 : tr1);
        ela.add_map(i20a, i21a, sign ? tr0 : tr1);
        ela.add_map(i21a, i30a, sign ? tr0 : tr1);
        ela.add_map(i22a, i23a, sign ? tr0 : tr1);
        ela.add_map(i23a, i28a, tr0);
        ela.add_map(i26a, i27a, sign ? tr0 : tr1);
        ela.add_map(i28a, i29a, sign ? tr0 : tr1);
        ela.add_map(i30a, i31a, sign ? tr0 : tr1);
        ela.mark_forbidden(i06a);
        ela.mark_forbidden(i07a);
        ela.mark_forbidden(i12a);
        ela.mark_forbidden(i13a);
        ela.mark_forbidden(i18a);
        ela.mark_forbidden(i19a);
        ela.mark_forbidden(i24a);
        ela.mark_forbidden(i25a);

        se3_t elb(bisb, mb, 2);
        libtensor::index<3> i00b, i01b, i02b, i03b, i04b, i05b, i06b, i07b;
        i04b[0] = 1; i03b[1] = 1; i03b[2] = 1; // 100
        i05b[0] = 1; i02b[1] = 1; i05b[2] = 1; // 101
        i06b[0] = 1; i06b[1] = 1; i01b[2] = 1; // 110
        i07b[0] = 1; i07b[1] = 1; i07b[2] = 1; // 111

        elb.add_map(i00b, i01b, sign ? tr0 : tr1);
        elb.add_map(i01b, i06b, sign ? tr0 : tr1);
        elb.add_map(i06b, i07b, sign ? tr0 : tr1);
        elb.mark_forbidden(i02b);
        elb.mark_forbidden(i03b);
        elb.mark_forbidden(i04b);
        elb.mark_forbidden(i05b);

        symmetry_element_set<5, double> seta(se5_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<3, double> setb_ref(se3_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<5> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
        sequence<5, size_t> seq(0); seq[2] = seq[3] = 1;
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<3>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Double merge of 4 dim of a 6-space on a 4-space.

    The partition symmetry in 6-space is constructed from two partition
    symmetries in 3-space:
    - Mappings:
      \code
      000 -> 011 (s1/s2)
      011 -> 101 (s1/s2)
      101 -> 110 (s1/s2)
      \endcode
    - Forbidden partitions: \c 001, \c 010, \c 100, \c 111

 **/
void so_merge_se_part_test::test_nmk_2(bool s1, bool s2) {

    std::ostringstream tnss;
    tnss << "so_merge_se_part_test::test_nmk_2(" << s1 << ", " << s2 << ")";

    typedef se_part<6, double> se6_t;
    typedef se_part<4, double> se4_t;
    typedef so_merge<6, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
    so_merge_se_t;

    try {

        libtensor::index<6> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5;
        i1b[3] = 5; i1b[4] = 5; i1b[5] = 5;
        block_index_space<6> bisa(dimensions<6>(index_range<6>(i1a, i1b)));
        mask<6> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        ma[3] = true; ma[4] = true; ma[5] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        libtensor::index<4> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5; i2b[2] = 5; i2b[3] = 5;
        block_index_space<4> bisb(dimensions<4>(index_range<4>(i2a, i2b)));
        mask<4> mb; mb[0] = true; mb[1] = true; mb[2] = true; mb[3] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se6_t ela(bisa, ma, 2);
        libtensor::index<6> ia[64];
        for (size_t i = 0; i < 64; i++) {
            size_t x = i;
            ia[i][0] = x / 32; x = x % 32;
            ia[i][1] = x / 16; x = x % 16;
            ia[i][2] = x /  8; x = x %  8;
            ia[i][3] = x /  4; x = x %  4;
            ia[i][4] = x /  2; x = x %  2;
            ia[i][5] = x;
        }
        scalar_transf<double> tr0, tr1(-1.);
        ela.add_map(ia[ 0], ia[ 3], s2 ? tr0 : tr1);
        ela.add_map(ia[ 3], ia[ 5], s2 ? tr0 : tr1);
        ela.add_map(ia[ 5], ia[ 6], s2 ? tr0 : tr1);
        ela.add_map(ia[ 6], ia[24], s1 == s2 ? tr0 : tr1);
        ela.add_map(ia[24], ia[27], s2 ? tr0 : tr1);
        ela.add_map(ia[27], ia[29], s2 ? tr0 : tr1);
        ela.add_map(ia[29], ia[30], s2 ? tr0 : tr1);
        ela.add_map(ia[30], ia[40], s1 == s2 ? tr0 : tr1);
        ela.add_map(ia[40], ia[43], s2 ? tr0 : tr1);
        ela.add_map(ia[43], ia[45], s2 ? tr0 : tr1);
        ela.add_map(ia[45], ia[46], s2 ? tr0 : tr1);
        ela.add_map(ia[46], ia[48], s1 == s2 ? tr0 : tr1);
        ela.add_map(ia[48], ia[51], s2 ? tr0 : tr1);
        ela.add_map(ia[51], ia[53], s2 ? tr0 : tr1);
        ela.add_map(ia[53], ia[54], s2 ? tr0 : tr1);

        ela.mark_forbidden(ia[ 1]); ela.mark_forbidden(ia[ 2]);
        ela.mark_forbidden(ia[ 4]); ela.mark_forbidden(ia[ 7]);
        ela.mark_forbidden(ia[ 8]); ela.mark_forbidden(ia[ 9]);
        ela.mark_forbidden(ia[10]); ela.mark_forbidden(ia[11]);
        ela.mark_forbidden(ia[12]); ela.mark_forbidden(ia[13]);
        ela.mark_forbidden(ia[14]); ela.mark_forbidden(ia[15]);
        ela.mark_forbidden(ia[16]); ela.mark_forbidden(ia[17]);
        ela.mark_forbidden(ia[18]); ela.mark_forbidden(ia[19]);
        ela.mark_forbidden(ia[20]); ela.mark_forbidden(ia[21]);
        ela.mark_forbidden(ia[22]); ela.mark_forbidden(ia[23]);
        ela.mark_forbidden(ia[25]); ela.mark_forbidden(ia[26]);
        ela.mark_forbidden(ia[28]); ela.mark_forbidden(ia[31]);
        ela.mark_forbidden(ia[32]); ela.mark_forbidden(ia[33]);
        ela.mark_forbidden(ia[34]); ela.mark_forbidden(ia[35]);
        ela.mark_forbidden(ia[36]); ela.mark_forbidden(ia[37]);
        ela.mark_forbidden(ia[38]); ela.mark_forbidden(ia[39]);
        ela.mark_forbidden(ia[41]); ela.mark_forbidden(ia[42]);
        ela.mark_forbidden(ia[44]); ela.mark_forbidden(ia[47]);
        ela.mark_forbidden(ia[49]); ela.mark_forbidden(ia[50]);
        ela.mark_forbidden(ia[52]); ela.mark_forbidden(ia[55]);
        ela.mark_forbidden(ia[56]); ela.mark_forbidden(ia[57]);
        ela.mark_forbidden(ia[58]); ela.mark_forbidden(ia[59]);
        ela.mark_forbidden(ia[60]); ela.mark_forbidden(ia[61]);
        ela.mark_forbidden(ia[62]); ela.mark_forbidden(ia[63]);

        se4_t elb(bisb, mb, 2);
        libtensor::index<4> i00b, i01b, i02b, i03b, i04b, i05b, i06b, i07b,
            i08b, i09b, i10b, i11b, i12b, i13b, i14b, i15b;
        i08b[0] = 1; i07b[1] = 1; i07b[2] = 1; i07b[3] = 1; // 1000
        i09b[0] = 1; i06b[1] = 1; i06b[2] = 1; i09b[3] = 1; // 1001
        i10b[0] = 1; i05b[1] = 1; i10b[2] = 1; i05b[3] = 1; // 1010
        i11b[0] = 1; i04b[1] = 1; i11b[2] = 1; i11b[3] = 1; // 1011
        i12b[0] = 1; i12b[1] = 1; i03b[2] = 1; i03b[3] = 1; // 1100
        i13b[0] = 1; i13b[1] = 1; i02b[2] = 1; i13b[3] = 1; // 1101
        i14b[0] = 1; i14b[1] = 1; i14b[2] = 1; i01b[3] = 1; // 1110
        i15b[0] = 1; i15b[1] = 1; i15b[2] = 1; i14b[3] = 1; // 1111

        elb.add_map(i00b, i06b, s1 == s2 ? tr0 : tr1);
        elb.add_map(i06b, i11b, s1 ? tr0 : tr1);
        elb.add_map(i11b, i13b, s1 == s2 ? tr0 : tr1);
        elb.mark_forbidden(i01b);
        elb.mark_forbidden(i02b);
        elb.mark_forbidden(i03b);
        elb.mark_forbidden(i04b);
        elb.mark_forbidden(i05b);
        elb.mark_forbidden(i07b);
        elb.mark_forbidden(i08b);
        elb.mark_forbidden(i09b);
        elb.mark_forbidden(i10b);
        elb.mark_forbidden(i12b);
        elb.mark_forbidden(i14b);
        elb.mark_forbidden(i15b);

        symmetry_element_set<6, double> seta(se6_t::k_sym_type);
        symmetry_element_set<4, double> setb(se4_t::k_sym_type);
        symmetry_element_set<4, double> setb_ref(se4_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<6> msk; msk[1] = msk[2] = msk[4] = msk[5] = true;
        sequence<6, size_t> seq(0); seq[2] = seq[5] = 1;
        symmetry_operation_params<so_merge_t> params(seta, msk, seq, setb);

        so_merge_se_t().perform(params);

        if(setb.is_empty()) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        compare_ref<4>::compare(tnss.str().c_str(), bisb, setb, setb_ref);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
