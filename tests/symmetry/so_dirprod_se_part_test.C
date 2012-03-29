#include <libtensor/symmetry/so_dirprod_se_part.h>
#include "../compare_ref.h"
#include "so_dirprod_se_part_test.h"

namespace libtensor {


void so_dirprod_se_part_test::perform() throw(libtest::test_exception) {

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
}


/**	\test Tests that the direct product of two empty group yields an empty
        group of a higher order
 **/
void so_dirprod_se_part_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname =
            "so_dirprod_se_part_test::test_empty_1()";

    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef se_part<5, double> se5_t;
    typedef so_dirprod<2, 3, double> so_t;
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


/**	\test Direct product of a group with mapping in 2-space and an empty
        group (1-space) forming a 3-space.
 **/
void so_dirprod_se_part_test::test_empty_2(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_se_part_test::test_empty_2(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirprod<2, 1, double> so_t;
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

        se2_t elema(bisa, ma, 2);
        elema.add_map(i00, i11, true);
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
            elemc.add_map(i000, i101, true);
            elemc.mark_forbidden(i001);
            elemc.mark_forbidden(i100);
        }
        else {
            elemc.add_map(i000, i110, true);
            elemc.mark_forbidden(i010);
            elemc.mark_forbidden(i100);
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

/**	\test Direct product of an empty group (1-space) and a group with mappings
        in 2-space forming a 3-space.
 **/
void so_dirprod_se_part_test::test_empty_3(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_se_part_test::test_empty_3(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirprod<1, 2, double> so_t;
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

        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i01, i11, false);
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
            elemc.add_map(i010, i110, false);
            elemc.mark_forbidden(i000);
            elemc.mark_forbidden(i100);
        }
        else {
            elemc.add_map(i001, i011, false);
            elemc.mark_forbidden(i000);
            elemc.mark_forbidden(i010);
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


/** \test Direct product of a group in 1-space and a group in 2-space.
 **/
void so_dirprod_se_part_test::test_nn_1(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_se_part_test::test_nn_1(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirprod<1, 2, double> so_t;
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
        se1_t elema(bisa, ma, 2);
        elema.add_map(i0, i1, symm1);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i00, i11, symm2);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        se3_t elemc(bisc, mc, 2);
        elemc.add_map(i000, i011, symm2);
        elemc.add_map(i011, i100, symm1 == symm2);
        elemc.add_map(i100, i111, symm2);
        elemc.mark_forbidden(i001);
        elemc.mark_forbidden(i010);
        elemc.mark_forbidden(i101);
        elemc.mark_forbidden(i110);

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

/** \test Direct product of a group in 1-space and a group in 2-space. The
        result is permuted with [012->120].
 **/
void so_dirprod_se_part_test::test_nn_2(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirprod_se_part_test::test_nn_2(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<1, double> se1_t;
    typedef se_part<2, double> se2_t;
    typedef se_part<3, double> se3_t;
    typedef so_dirprod<1, 2, double> so_t;
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
        se1_t elema(bisa, ma, 2);
        elema.add_map(i0, i1, symm1);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        se2_t elemb(bisb, mb, 2);
        elemb.add_map(i00, i11, symm2);
        elemb.mark_forbidden(i01);
        elemb.mark_forbidden(i10);

        index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;

        se3_t elemc(bisc, mc, 2);
        elemc.add_map(i000, i001, symm1);
        elemc.add_map(i001, i110, symm1 == symm2);
        elemc.add_map(i110, i111, symm1);
        elemc.mark_forbidden(i010);
        elemc.mark_forbidden(i011);
        elemc.mark_forbidden(i100);
        elemc.mark_forbidden(i101);

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



} // namespace libtensor
