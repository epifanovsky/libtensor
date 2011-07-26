#include <libtensor/symmetry/part/so_merge_impl_part.h>
#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_merge_impl_part_test.h"


namespace libtensor {

void so_merge_impl_part_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2();
    test_nm1_1(true); test_nm1_1(false);
    test_nm1_2(true); test_nm1_2(false);
    test_2n2nn_1(true, true); test_2n2nn_1(false, true);
    test_2n2nn_1(true, false); test_2n2nn_1(false, false);
    test_2n2nn_2(true, true); test_2n2nn_2(false, true);
    test_2n2nn_2(true, false); test_2n2nn_2(false, false);
}


/**	\test Tests that a single merge of 2 dim of an empty partition set yields
        an empty partition set of lower order
 **/
void so_merge_impl_part_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname = "so_merge_impl_part_test::test_empty_1()";

    typedef se_part<4, double> se4_t;
    typedef se_part<3, double> se3_t;
    typedef so_merge<4, 2, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
    so_merge_impl_t;

    try {

        symmetry_element_set<4, double> set1(se4_t::k_sym_type);
        symmetry_element_set<3, double> set2(se3_t::k_sym_type);

        mask<4> msk[1]; msk[0][2] = true; msk[0][3] = true;
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

/** \test Tests that a double merge of dimensions of an empty partition set
        yields an empty partition set of lower order
 **/
void so_merge_impl_part_test::test_empty_2() throw(libtest::test_exception) {

    static const char *testname = "so_merge_impl_part_test::test_empty_2()";

    typedef se_part<5, double> se5_t;
    typedef se_part<3, double> se3_t;
    typedef so_merge<5, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se5_t>
    so_merge_impl_t;

    try {

        symmetry_element_set<5, double> set1(se5_t::k_sym_type);
        symmetry_element_set<3, double> set2(se3_t::k_sym_type);

        mask<5> msk[2];
        msk[0][0] = true; msk[0][2] = true;
        msk[1][1] = true; msk[1][3] = true;
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

/**	\test Single merge of 2 dim of a 3-space on a 2-space.
 **/
void so_merge_impl_part_test::test_nm1_1(bool sign)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_part_test::test_nm1_1(" << sign << ")";

    typedef se_part<3, double> se3_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<3, 2, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t>
    so_merge_impl_t;

    try {

        index<3> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5;
        block_index_space<3> bisa(dimensions<3>(index_range<3>(i1a, i1b)));
        mask<3> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb;
        mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se3_t ela(bisa, ma, 2);
        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        ela.add_map(i000, i001, true);
        ela.add_map(i001, i110, sign);
        ela.add_map(i110, i111, true);
        ela.mark_forbidden(i010);
        ela.mark_forbidden(i011);
        ela.mark_forbidden(i100);
        ela.mark_forbidden(i101);

        se2_t elb(bisb, mb, 2);
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        elb.add_map(i00, i11, sign);
        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<3> mc[1]; mc[0][0] = true; mc[0][2] = true;
        symmetry_operation_params<so_merge_t> params(seta, mc, setb);
        so_merge_impl_t().perform(params);

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
void so_merge_impl_part_test::test_nm1_2(bool sign)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_part_test::test_nm1_1(" << tnss << ")";

    typedef se_part<3, double> se3_t;
    typedef se_part<1, double> se1_t;
    typedef so_merge<3, 3, 1, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se3_t>
    so_merge_impl_t;

    try {

        index<3> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5;
        block_index_space<3> bisa(dimensions<3>(index_range<3>(i1a, i1b)));
        mask<3> ma;
        ma[0] = true; ma[1] = true; ma[2] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<1> i2a, i2b;
        i2b[0] = 5;
        block_index_space<1> bisb(dimensions<1>(index_range<1>(i2a, i2b)));
        mask<1> mb;
        mb[0] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se3_t ela(bisa, ma, 2);
        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        ela.add_map(i000, i001, true);
        ela.add_map(i001, i110, sign);
        ela.add_map(i110, i111, true);
        ela.mark_forbidden(i010);
        ela.mark_forbidden(i011);
        ela.mark_forbidden(i100);
        ela.mark_forbidden(i101);

        se1_t elb(bisb, mb, 2);
        index<1> i0, i1;
        i1[0] = 1;
        elb.add_map(i0, i1, sign);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<1, double> setb(se1_t::k_sym_type);
        symmetry_element_set<1, double> setb_ref(se1_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<3> mc[1];
        mc[0][0] = true; mc[0][1] = true; mc[0][2] = true;
        symmetry_operation_params<so_merge_t> params(seta, mc, setb);
        so_merge_impl_t().perform(params);

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
void so_merge_impl_part_test::test_2n2nn_1(bool s1, bool s2)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_part_test::test_2n2nn_1(" << s1 << ", " << s2 << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<4, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
    so_merge_impl_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma; ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se4_t ela(bisa, ma, 2);
        index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        ela.add_map(i0000, i0011, s2);
        ela.add_map(i0011, i1100, s1 == s2);
        ela.add_map(i1100, i1111, s2);
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
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        elb.add_map(i00, i11, s1 == s2);
        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> mc[2];
        mc[0][0] = true; mc[0][2] = true;
        mc[1][1] = true; mc[1][3] = true;
        symmetry_operation_params<so_merge_t> params(seta, mc, setb);
        so_merge_impl_t().perform(params);

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
void so_merge_impl_part_test::test_2n2nn_2(bool s1, bool s2)
throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_merge_impl_part_test::test_2n2nn_2(" << s1 << ", " << s2 << ")";

    typedef se_part<4, double> se4_t;
    typedef se_part<2, double> se2_t;
    typedef so_merge<4, 4, 2, double> so_merge_t;
    typedef symmetry_operation_impl<so_merge_t, se4_t>
    so_merge_impl_t;

    try {

        index<4> i1a, i1b;
        i1b[0] = 5; i1b[1] = 5; i1b[2] = 5; i1b[3] = 5;
        block_index_space<4> bisa(dimensions<4>(index_range<4>(i1a, i1b)));
        mask<4> ma; ma[0] = true; ma[1] = true; ma[2] = true; ma[3] = true;
        bisa.split(ma, 2);
        bisa.split(ma, 3);
        bisa.split(ma, 5);

        index<2> i2a, i2b;
        i2b[0] = 5; i2b[1] = 5;
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i2a, i2b)));
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 2);
        bisb.split(mb, 3);
        bisb.split(mb, 5);

        se4_t ela(bisa, ma, 2);
        index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
        i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        if (s1) {
            ela.add_map(i0000, i1100, s1);
            ela.add_map(i0001, i1101, s1);
            ela.add_map(i0010, i1110, s1);
            ela.add_map(i0011, i1111, s1);
        }
        if (s2) {
            ela.add_map(i0000, i0011, s2);
            ela.add_map(i0100, i0111, s2);
            ela.add_map(i1000, i1011, s2);
            ela.add_map(i1100, i1111, s2);
        }
        if (s1 == s2 && ! s1) {
            ela.add_map(i0000, i1111, s1);
            ela.add_map(i0011, i1100, s1);
        }
        ela.mark_forbidden(i0101);
        ela.mark_forbidden(i0110);
        ela.mark_forbidden(i1001);
        ela.mark_forbidden(i1010);

        se2_t elb(bisb, mb, 2);
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        if (s1 == s2)
            elb.add_map(i00, i11, s1);

        elb.mark_forbidden(i01);
        elb.mark_forbidden(i10);

        symmetry_element_set<4, double> seta(se4_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb_ref(se2_t::k_sym_type);

        seta.insert(ela);
        setb_ref.insert(elb);

        mask<4> mc[2];
        mc[0][0] = true; mc[0][2] = true;
        mc[1][1] = true; mc[1][3] = true;
        symmetry_operation_params<so_merge_t> params(seta, mc, setb);
        so_merge_impl_t().perform(params);

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
