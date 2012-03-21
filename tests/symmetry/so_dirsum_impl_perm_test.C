#include <libtensor/symmetry/so_dirsum_se_perm.h>
#include "../compare_ref.h"
#include "so_dirsum_impl_perm_test.h"

namespace libtensor {


void so_dirsum_impl_perm_test::perform() throw(libtest::test_exception) {

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


/**	\test Tests that the direct sum of two empty group yields an empty
        group of a higher order.
 **/
void so_dirsum_impl_perm_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname =
            "so_dirsum_impl_perm_test::test_empty_1()";

    typedef se_perm<2, double> se2_t;
    typedef se_perm<3, double> se3_t;
    typedef se_perm<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1, i2;
        i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
        block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
        mask<5> m;
        m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
        bis.split(m, 1);
        bis.split(m, 2);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        permutation<5> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bis, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(testname, bis, setc, setc_ref);

        if(! setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Direct sum of a group with one element [012->120](+/-) and an
        empty group (2-space) forming a 5-space. Expected is a group
        containing a single element (+) or an empty group (-).
 **/
void so_dirsum_impl_perm_test::test_empty_2(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_perm_test::test_empty_2(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_perm<2, double> se2_t;
    typedef se_perm<3, double> se3_t;
    typedef se_perm<5, double> se5_t;
    typedef so_dirsum<3, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1, i2;
        i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
        block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
        mask<5> m;
        m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
        bis.split(m, 1);
        bis.split(m, 2);

        permutation<3> p1; p1.permute(0, 1).permute(1, 2);
        permutation<5> p2;
        if (perm) p2.permute(1, 4).permute(0, 1);
        else p2.permute(0, 1).permute(1, 2);
        se3_t elema(p1, true);
        se5_t elemc(p2, true);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        seta.insert(elema);
        setc_ref.insert(elemc);

        permutation<5> px;
        if (perm) px.permute(0, 2).permute(2, 4);
        symmetry_operation_params<so_t> params(seta, setb, px, bis, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(tns.c_str(), bis, setc, setc_ref);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/**	\test Direct product of an empty group (2-space) and a group with one
        element [012->201](+/-) forming a 5-space. Expected is a group
        containing a single element (+) or an empty group (-).
 **/
void so_dirsum_impl_perm_test::test_empty_3(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_perm_test::test_empty_3(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_perm<2, double> se2_t;
    typedef se_perm<3, double> se3_t;
    typedef se_perm<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1, i2;
        i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
        block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
        mask<5> m;
        m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
        bis.split(m, 1);
        bis.split(m, 2);

        permutation<3> p1; p1.permute(1, 2).permute(0, 1);
        permutation<5> p2;
        if (perm) p2.permute(0, 2).permute(2, 3);
        else p2.permute(3, 4).permute(2, 3);
        se3_t elemb(p1, true);
        se5_t elemc(p2, true);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        setb.insert(elemb);
        setc_ref.insert(elemc);

        permutation<5> px;
        if (perm) px.permute(0, 2).permute(2, 4);
        symmetry_operation_params<so_t> params(seta, setb, px, bis, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(tns.c_str(), bis, setc, setc_ref);

        if (setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Direct product of a group of one element [01->10](+/-) and a group
        with two elements [012->120] and [01->10](+/-) forming
        a 5-space.
 **/
void so_dirsum_impl_perm_test::test_nn_1(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_perm_test::test_nn_1(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_perm<2, double> se2_t;
    typedef se_perm<3, double> se3_t;
    typedef se_perm<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1, i2;
        i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
        block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
        mask<5> m;
        m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
        bis.split(m, 1);
        bis.split(m, 2);

        se2_t elema(permutation<2>().permute(0, 1), symm1);
        se3_t elemb1(permutation<3>().permute(0, 1).permute(1, 2), true);
        se3_t elemb2(permutation<3>().permute(0, 1), symm2);
        se5_t elemc1(permutation<5>().permute(0, 1), symm1);
        se5_t elemc2(permutation<5>().permute(2, 3).permute(3, 4), true);
        se5_t elemc3(permutation<5>().permute(2, 3), symm2);
        se5_t elemc4(permutation<5>().permute(0, 1).permute(2, 3), false);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb1);
        setb.insert(elemb2);

        setc_ref.insert(elemc2);
        if (symm1) setc_ref.insert(elemc1);
        if (symm2) setc_ref.insert(elemc3);
        if (! symm1 && ! symm2) setc_ref.insert(elemc4);

        permutation<5> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bis, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(tns.c_str(), bis, setc, setc_ref);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Direct product of a group with two elements [012->120](+) and
        [01->10](+/-) and a group of one element [01->10](+/-) forming
        a 5-space. The result is permuted with [01234->13204].
 **/
void so_dirsum_impl_perm_test::test_nn_2(
        bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_perm_test::test_nn_2(" << symm1 << ", "
            << symm2 << ")";
    std::string tns = tnss.str();

    typedef se_perm<2, double> se2_t;
    typedef se_perm<3, double> se3_t;
    typedef se_perm<5, double> se5_t;
    typedef so_dirsum<3, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1, i2;
        i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2; i2[4] = 2;
        block_index_space<5> bis(dimensions<5>(index_range<5>(i1, i2)));
        mask<5> m;
        m[0] = true; m[1] = true; m[2] = true; m[3] = true; m[4] = true;
        bis.split(m, 1);
        bis.split(m, 2);

        se3_t elema1(permutation<3>().permute(0, 1).permute(1, 2), true);
        se3_t elema2(permutation<3>().permute(0, 1), symm1);
        se2_t elemb(permutation<2>().permute(0, 1), symm2);
        se5_t elemc1(permutation<5>().permute(0, 2).permute(2, 3), true);
        se5_t elemc2(permutation<5>().permute(0, 3), symm1);
        se5_t elemc3(permutation<5>().permute(1, 4), symm2);
        se5_t elemc4(permutation<5>().permute(0, 3).permute(1, 4), false);

        symmetry_element_set<3, double> seta(se3_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        seta.insert(elema1);
        seta.insert(elema2);
        setb.insert(elemb);

        setc_ref.insert(elemc1);
        if (symm1) setc_ref.insert(elemc2);
        if (symm2) setc_ref.insert(elemc3);
        if (! symm1 && ! symm2) setc_ref.insert(elemc4);

        permutation<5> px;
        px.permute(0, 1).permute(1, 3);
        symmetry_operation_params<so_t> params(seta, setb, px, bis, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(tns.c_str(), bis, setc, setc_ref);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}



} // namespace libtensor
