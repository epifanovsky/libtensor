#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/combine_part.h>
#include "../compare_ref.h"
#include "combine_part_test.h"

namespace libtensor {


void combine_part_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2(true);
    test_2(false);
    test_3(true, true);
    test_3(true, false);
    test_3(false, true);
    test_3(false, false);
    test_4a(true, true, true);
    test_4a(true, false, true);
    test_4a(false, true, true);
    test_4a(false, false, true);
    test_4a(true, true, false);
    test_4a(true, false, false);
    test_4a(false, true, false);
    test_4a(false, false, false);
    test_4b(true, true);
    test_4b(true, false);
    test_4b(false, true);
    test_4b(false, false);
}


/** \test Tests that calling combine part on an empty set throws an exception
 **/
void combine_part_test::test_1() throw(libtest::test_exception) {

    static const char *testname =
            "combine_part_test::test_1()";

    typedef se_part<2, double> se2_t;
    typedef combine_part<2, double> combine_t;

    index<2> i1c, i2c;
    i2c[0] = 3; i2c[1] = 3;
    block_index_space<2> bisc(dimensions<2>(index_range<2>(i1c, i2c)));

    mask<2> mc; mc[0] = true; mc[1] = true;
    bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

    symmetry_element_set<2, double> seta(se2_t::k_sym_type);

    bool exc = false;
    try {

        combine_t comb(seta);

    } catch (exception &e) {
        exc = true;
    }

    if (! exc) {
        fail_test(testname, __FILE__, __LINE__, "No exception.");
    }
}


/** \test Combine a set with a single element in 2-space
 **/
void combine_part_test::test_2(bool symm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "combine_part_test::test_2(" << symm << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se_t;
    typedef combine_part<2, double> combine_t;

    try {

        index<2> i1, i2; i2[0] = 3; i2[1] = 3;

        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m; m[0] = true; m[1] = true;
        bis.split(m, 1); bis.split(m, 2); bis.split(m, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;

        se_t elem_ref(bis, m, 2);
        scalar_transf<double> tr1(2.0), tr2(0.5);
        elem_ref.add_map(i00, i11, tr1);
        elem_ref.mark_forbidden(i01);
        elem_ref.mark_forbidden(i10);

        symmetry_element_set<2, double> set(se_t::k_sym_type);

        set.insert(elem_ref);

        combine_t comb(set);
        se_t elem(comb.get_bis(), comb.get_pdims());
        comb.perform(elem);

        if (! elem.map_exists(i00, i11)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Map i00->i11 missing.");
        }
        if (elem.get_transf(i00, i11) != tr1) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Wrong transformation at i00->i11.");
        }
        if (elem.get_transf(i11, i00) != tr2) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Wrong transformation at i11->i00.");
        }
        if (! elem.is_forbidden(i01)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "i01 not forbidden.");
        }
        if (! elem.is_forbidden(i10)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "i10 not forbidden.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Combine a set with 2 partitions in 1-space to a partition in 2-space
 **/
void combine_part_test::test_3(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "combine_part_test::test_3(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<3, double> se_t;
    typedef combine_part<3, double> combine_t;

    try {

        index<3> i1, i2; i2[0] = 3; i2[1] = 3; i2[2] = 3;
        index<3> i1p, i2p; i2p[0] = 3; i2p[1] = 1; i2p[2] = 1;

        block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
        mask<3> m; m[0] = true; m[1] = true; m[2] = true;
        bis.split(m, 1); bis.split(m, 2); bis.split(m, 3);

        dimensions<3> pdims(index_range<3>(i1p, i2p));

        index<3> i000, i001, i010, i011, i100, i101, i110, i111,
        i200, i201, i210, i211, i300, i301, i310, i311;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        i200[0] = i201[0] = i210[0] = i211[0] = 2;
        i210[1] = 1; i201[2] = 1;
        i211[1] = 1; i211[2] = 1;
        i300[0] = i301[0] = i310[0] = i311[0] = 3;
        i310[1] = 1; i301[2] = 1;
        i311[1] = 1; i311[2] = 1;

        mask<3> ma, mb; ma[0] = true; mb[1] = true; mb[2] = true;
        se_t elema(bis, ma, 4), elemb(bis, mb, 2), elem_ref(bis, pdims);
        scalar_transf<double> tr1(symm1 ? 1.0 : -1.0), tr2(symm2 ? 1.0 : -1.0);
        elema.add_map(i000, i200, tr1);
        elema.mark_forbidden(i100);
        elema.mark_forbidden(i300);
        elemb.add_map(i000, i011, tr2);
        elemb.mark_forbidden(i001);
        elemb.mark_forbidden(i010);

        elem_ref.add_map(i000, i011, tr2);
        elem_ref.add_map(i000, i200, tr1);
        elem_ref.add_map(i200, i211, tr2);
        elem_ref.mark_forbidden(i001);
        elem_ref.mark_forbidden(i010);
        elem_ref.mark_forbidden(i100);
        elem_ref.mark_forbidden(i101);
        elem_ref.mark_forbidden(i110);
        elem_ref.mark_forbidden(i111);
        elem_ref.mark_forbidden(i201);
        elem_ref.mark_forbidden(i210);
        elem_ref.mark_forbidden(i300);
        elem_ref.mark_forbidden(i301);
        elem_ref.mark_forbidden(i310);
        elem_ref.mark_forbidden(i311);

        symmetry_element_set<3, double> set1(se_t::k_sym_type);
        symmetry_element_set<3, double> set2(se_t::k_sym_type);
        symmetry_element_set<3, double> set_ref(se_t::k_sym_type);

        set1.insert(elema);
        set1.insert(elemb);
        set_ref.insert(elem_ref);

        combine_t comb(set1);
        se_t elem(comb.get_bis(), comb.get_pdims());
        comb.perform(elem);

        set2.insert(elem);

        compare_ref<3>::compare(tns.c_str(), bis, set2, set1);
        compare_ref<3>::compare(tns.c_str(), bis, set2, set_ref);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Combine two partitions in 2-space into 1 partition in 2-space
 **/
void combine_part_test::test_4a(bool symm1, bool symm2,
        bool forbidden) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "combine_part_test::test_4a(" << symm1 << ", "
            << symm2 << ", " << forbidden << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se_t;
    typedef combine_part<2, double> combine_t;

    try {

        index<2> i1, i2; i2[0] = 3; i2[1] = 3;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));

        mask<2> m; m[0] = true; m[1] = true;
        bis.split(m, 1); bis.split(m, 2); bis.split(m, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        se_t elema(bis, m, 2), elemb(bis, m, 2), elem_ref(bis, m, 2);
        scalar_transf<double> tr1(symm1 ? 1. : -1.), tr2(symm2 ? 1. : -1.);
        elema.add_map(i00, i11, tr1);
        elema.add_map(i01, i10, tr1);
        elemb.add_map(i00, i11, tr2);

        if (symm1 == symm2)
            elem_ref.add_map(i00, i11, tr1);
        else {
            elem_ref.mark_forbidden(i00);
            elem_ref.mark_forbidden(i11);
        }
        elem_ref.add_map(i01, i10, tr1);

        if (forbidden) {
            elemb.mark_forbidden(i01);
            elemb.mark_forbidden(i10);
            elem_ref.mark_forbidden(i01);
            elem_ref.mark_forbidden(i10);
        }

        symmetry_element_set<2, double> set1(se_t::k_sym_type);
        symmetry_element_set<2, double> set2(se_t::k_sym_type);
        symmetry_element_set<2, double> set_ref(se_t::k_sym_type);

        set1.insert(elema);
        set1.insert(elemb);
        set_ref.insert(elem_ref);

        combine_t comb(set1);
        se_t elem(comb.get_bis(), comb.get_pdims());
        comb.perform(elem);

        set2.insert(elem);

        compare_ref<2>::compare(tns.c_str(), bis, set2, set_ref);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Combine two partitions in 2-space into 1 partition in 2-space
 **/
void combine_part_test::test_4b(bool symm1,
        bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "combine_part_test::test_4b(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_part<2, double> se_t;
    typedef combine_part<2, double> combine_t;

    try {

        index<2> i1, i2; i2[0] = 3; i2[1] = 3;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));

        mask<2> m; m[0] = true; m[1] = true;
        bis.split(m, 1); bis.split(m, 2); bis.split(m, 3);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr1(symm1 ? 1. : -1.), tr2(symm2 ? 1. : -1.);
        se_t elema(bis, m, 2), elemb(bis, m, 2), elem_ref(bis, m, 2);
        elema.add_map(i00, i11, tr1);
        elemb.add_map(i00, i01, tr2);
        elemb.add_map(i10, i11, tr2);
        elem_ref.add_map(i00, i01, tr2);
        elem_ref.add_map(i01, i10, tr1);
        elem_ref.add_map(i10, i11, tr2);

        symmetry_element_set<2, double> set1(se_t::k_sym_type);
        symmetry_element_set<2, double> set2(se_t::k_sym_type);
        symmetry_element_set<2, double> set_ref(se_t::k_sym_type);

        set1.insert(elema);
        set1.insert(elemb);
        set_ref.insert(elem_ref);

        combine_t comb(set1);
        se_t elem(comb.get_bis(), comb.get_pdims());
        comb.perform(elem);

        set2.insert(elem);

        compare_ref<2>::compare(tns.c_str(), bis, set2, set1);
        compare_ref<2>::compare(tns.c_str(), bis, set2, set_ref);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
