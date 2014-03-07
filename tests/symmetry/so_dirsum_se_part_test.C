#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_dirsum_se_part.h>
#include "../compare_ref.h"
#include "so_dirsum_se_part_test.h"

namespace libtensor {


void so_dirsum_se_part_test::perform() throw(libtest::test_exception) {

    test_empty_1();
    test_empty_2(true);
    test_empty_2(false);
    test_empty_3(true);
    test_empty_3(false);
    test_nn_1(true, true);
    test_nn_1(true, false);
    test_nn_1(false, true);
    test_nn_1(false, false);
    test_nn_2(true, true);
    test_nn_2(true, false);
    test_nn_2(false, true);
    test_nn_2(false, false);
    test_nn_3(true, true);
    test_nn_3(true, false);
    test_nn_3(false, true);
    test_nn_3(false, false);
    test_nn_4(true, true);
    test_nn_4(true, false);
    test_nn_4(false, true);
    test_nn_4(false, false);
    test_nn_5(true);
    test_nn_5(false);
}


/** \test Tests that the direct sum of two empty group yields an empty
        group of a higher order
 **/
void so_dirsum_se_part_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname =
            "so_dirsum_se_part_test::test_empty_1()";

    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef se_part<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_se_t;

    try {

        index<5> i1c, i2c;
        i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3; i2c[4] = 3;
        block_index_space<5> bisc(dimensions<5>(index_range<5>(i1c, i2c)));

        mask<5> mc;
        mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true; mc[4] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        permutation<5> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<5>::compare(testname, bisc, setc, setc_ref);

        if(! setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of a group with mapping in 2-space and an empty
        group (1-space) forming a 3-space.
 **/
void so_dirsum_se_part_test::test_empty_2(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_empty_2(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirsum<2, 1, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_se_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0;

        se2_t elema(bisa, ma, 2);
        elema.add_map(i00, i11, tr0);
        elema.mark_forbidden(i01);
        elema.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        mask<3> mx;
        if (perm) { mx[0] = true; mx[2] = true; }
        else { mx[0] = true; mx[1] = true; }

        se3_t elemc(bisc, mx, 2);
        if (perm) {
            elemc.add_map(i000, i101, tr0);
            elemc.add_map(i001, i100, tr0);
        }
        else {
            elemc.add_map(i000, i110, tr0);
            elemc.add_map(i010, i100, tr0);
        }

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<1, double> setb(se1_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);
        symmetry_element_set<3, double> setc_ref(se3_t::k_sym_type);

        seta.insert(elema);
        setc_ref.insert(elemc);

        permutation<3> px;
        if (perm) px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<3>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of an empty group (1-space) and a group with mappings
        in 2-space forming a 3-space.
 **/
void so_dirsum_se_part_test::test_empty_3(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_empty_3(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_se_t;

    try {

        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);

        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i01, i11, tr1);
        elemb.mark_forbidden(i00);
        elemb.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        mask<3> mx;
        if (perm) { mx[0] = true; mx[1] = true; }
        else { mx[1] = true; mx[2] = true; }

        se3_t elemc(bisc, mx, 2);
        if (perm) {
            elemc.add_map(i000, i100, tr0);
        }
        else {
            elemc.add_map(i000, i010, tr0);
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);
        symmetry_element_set<3, double> setc_ref(se3_t::k_sym_type);

        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<3> px;
        if (perm) px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<3>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of a group in 1-space and a group in 2-space.
 **/
void so_dirsum_se_part_test::test_nn_1(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_nn_1(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_se_t;

    try {

        index<1> i1a, i2a; i2a[0] = 3; ;
        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<1> ma; ma[0] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<1> i0, i1; i1[0] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        se1_t elema(bisa, ma, 2);
        elema.add_map(i0, i1, symm1 ? tr0 : tr1);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i00, i11, symm2 ? tr0 : tr1);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        se3_t elemc(bisc, mc, 2);
        elemc.add_map(i001, i010, tr0);
        elemc.add_map(i101, i110, tr0);
        elemc.add_map(i001, i101, symm1 ? tr0 : tr1);

        if (symm1) {
            elemc.add_map(i000, i100, symm1 ? tr0 : tr1);
            elemc.add_map(i011, i111, symm1 ? tr0 : tr1);
        }
        if (symm2) {
            elemc.add_map(i000, i011, symm2 ? tr0 : tr1);
            elemc.add_map(i100, i111, symm2 ? tr0 : tr1);
        }
        if (symm1 == symm2) {
            elemc.add_map(i000, i111, symm1 ? tr0 : tr1);
            elemc.add_map(i011, i100, symm1 ? tr0 : tr1);
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);
        symmetry_element_set<3, double> setc_ref(se3_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<3> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<3>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of a group in 1-space and a group in 2-space. The
        result is permuted with [012->120].
 **/
void so_dirsum_se_part_test::test_nn_2(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_nn_2(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_se_t;

    try {

        index<1> i1a, i2a; i2a[0] = 3; ;
        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<1> ma; ma[0] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<1> i0, i1; i1[0] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        se1_t elema(bisa, ma, 2);
        elema.add_map(i0, i1, symm1 ? tr0 : tr1);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i00, i11, symm2 ? tr0 : tr1);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        se3_t elemc(bisc, mc, 2);
        elemc.add_map(i010, i100, tr0);
        elemc.add_map(i011, i101, tr0);
        elemc.add_map(i010, i011, symm1 ? tr0 : tr1);

        if (symm1) {
            elemc.add_map(i000, i001, symm1 ? tr0 : tr1);
            elemc.add_map(i110, i111, symm1 ? tr0 : tr1);
        }
        if (symm2) {
            elemc.add_map(i000, i110, symm2 ? tr0 : tr1);
            elemc.add_map(i001, i111, symm2 ? tr0 : tr1);
        }
        if (symm1 == symm2) {
            elemc.add_map(i000, i111, symm1 ? tr0 : tr1);
            elemc.add_map(i110, i001, symm1 ? tr0 : tr1);
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);
        symmetry_element_set<3, double> setc_ref(se3_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<3> px;
        px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<3>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of two groups in 2-space.
 **/
void so_dirsum_se_part_test::test_nn_3(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_nn_3(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_dirsum<2, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se4_t> so_se_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<4> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        se2_t elema(bisa, ma, 2), elemb(bisa, ma, 2);
        elema.add_map(i01, i10, symm1 ? tr0 : tr1);
        elema.mark_forbidden(i00);

        elemb.add_map(i00, i11, symm2 ? tr0 : tr1);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);

        index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111;
        index<4> i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        se4_t elemc(bisc, mc, 2);
        elemc.add_map(i0000, i0011, symm2 ? tr0 : tr1);
        elemc.add_map(i0101, i0110, tr0);
        elemc.add_map(i1001, i1010, tr0);
        elemc.add_map(i1101, i1110, tr0);
        elemc.add_map(i0101, i1001, symm1 ? tr0 : tr1);
        elemc.mark_forbidden(i0001);
        elemc.mark_forbidden(i0010);

        if (symm1) {
            elemc.add_map(i0100, i1000, symm1 ? tr0 : tr1);
            elemc.add_map(i0111, i1011, symm1 ? tr0 : tr1);
        }
        if (symm2) {
            elemc.add_map(i0100, i0111, symm2 ? tr0 : tr1);
            elemc.add_map(i1000, i1011, symm2 ? tr0 : tr1);
            elemc.add_map(i1100, i1111, symm2 ? tr0 : tr1);
        }
        if (symm1 == symm2) {
            elemc.add_map(i0100, i1011, symm1 ? tr0 : tr1);
            elemc.add_map(i0111, i1000, symm1 ? tr0 : tr1);
        }

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<4, double> setc(se4_t::k_sym_type);
        symmetry_element_set<4, double> setc_ref(se4_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<4> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<4>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of two groups in 2-space.
 **/
void so_dirsum_se_part_test::test_nn_4(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_nn_4(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_dirsum<2, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se4_t> so_se_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<4> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        se2_t elema(bisa, ma, 2), elemb(bisa, ma, 2);
        elema.add_map(i00, i11, symm1 ? tr0 : tr1);
        elema.add_map(i01, i10, symm1 ? tr0 : tr1);

        elemb.add_map(i00, i11, symm2 ? tr0 : tr1);
        elemb.add_map(i01, i10, symm2 ? tr0 : tr1);

        index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111;
        index<4> i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        se4_t elemc(bisc, mc, 2);
        if (symm1) {
            elemc.add_map(i0000, i1010, tr0);
            elemc.add_map(i0001, i1011, tr0);
            elemc.add_map(i0100, i1110, tr0);
            elemc.add_map(i0101, i1111, tr0);

            elemc.add_map(i0010, i1000, tr0);
            elemc.add_map(i0011, i1001, tr0);
            elemc.add_map(i0110, i1100, tr0);
            elemc.add_map(i0111, i1101, tr0);
        }
        if (symm2) {
            elemc.add_map(i0000, i0101, tr0);
            elemc.add_map(i0010, i0111, tr0);
            elemc.add_map(i1000, i1101, tr0);
            elemc.add_map(i1010, i1111, tr0);

            elemc.add_map(i0001, i0100, tr0);
            elemc.add_map(i0011, i0110, tr0);
            elemc.add_map(i1001, i1100, tr0);
            elemc.add_map(i1011, i1110, tr0);
        }

        if (! symm1 && symm1 == symm2) {
            elemc.add_map(i0000, i1111, tr1);
            elemc.add_map(i0001, i1110, tr1);
            elemc.add_map(i0100, i1011, tr1);
            elemc.add_map(i0101, i1010, tr1);
            elemc.add_map(i0010, i1101, tr1);
            elemc.add_map(i0011, i1100, tr1);
            elemc.add_map(i0110, i1001, tr1);
            elemc.add_map(i0111, i1000, tr1);
        }

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<4, double> setc(se4_t::k_sym_type);
        symmetry_element_set<4, double> setc_ref(se4_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<4> px; px.permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<4>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of two groups in 2-space.
 **/
void so_dirsum_se_part_test::test_nn_5(
        bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_se_part_test::test_nn_5(" << symm << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se2_t;
    typedef se_part<4, double> se4_t;
    typedef so_dirsum<2, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se4_t> so_se_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<4> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);
        se2_t elema(bisa, ma, 2), elemb(bisa, ma, 2);
        elema.add_map(i00, i11, symm ? tr0 : tr1);
        elema.mark_forbidden(i01);
        elema.mark_forbidden(i10);

        elemb.mark_forbidden(i00);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);
        elemb.mark_forbidden(i11);

        index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111;
        index<4> i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        se4_t elemc(bisc, mc, 2);
        elemc.add_map(i0000, i0001, tr0);
        elemc.add_map(i0001, i0100, tr0);
        elemc.add_map(i0100, i0101, tr0);
        elemc.add_map(i0101, i1010, symm ? tr0 : tr1);
        elemc.add_map(i1010, i1011, tr0);
        elemc.add_map(i1011, i1110, tr0);
        elemc.add_map(i1110, i1111, tr0);
        elemc.mark_forbidden(i0010);
        elemc.mark_forbidden(i0011);
        elemc.mark_forbidden(i0110);
        elemc.mark_forbidden(i0111);
        elemc.mark_forbidden(i1000);
        elemc.mark_forbidden(i1001);
        elemc.mark_forbidden(i1100);
        elemc.mark_forbidden(i1101);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<4, double> setc(se4_t::k_sym_type);
        symmetry_element_set<4, double> setc_ref(se4_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<4> px; px.permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_se_t().perform(params);

        compare_ref<4>::compare(tns.c_str(), bisc, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
