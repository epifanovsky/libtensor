#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_apply_se_perm.h>
#include "so_apply_se_perm_test.h"

namespace libtensor {


void so_apply_se_perm_test::perform() throw(libtest::test_exception) {

    test_1(false, false, false);
    test_1(false, false,  true);
    test_1(false,  true, false);
    test_1( true, false, false);
    test_1( true, false,  true);
    test_1( true,  true, false);
    test_2(false, false, false);
    test_2(false, false,  true);
    test_2(false,  true, false);
    test_2( true, false, false);
    test_2( true, false,  true);
    test_2( true,  true, false);
    test_3(false, false, false);
    test_3(false, false,  true);
    test_3(false,  true, false);
    test_3( true, false, false);
    test_3( true, false,  true);
    test_3( true,  true, false);
}


/** \test Tests that an empty sets yields an empty set
 **/
void so_apply_se_perm_test::test_1(bool keep_zero,
        bool is_asym, bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_apply_se_perm_test::test_1(" << keep_zero << ", "
            << is_asym << ", " << sign << ")";

    typedef se_perm<2, double> se_t;
    typedef so_apply<2, double> so_t;
    typedef symmetry_operation_params<so_t> params_t;
    typedef symmetry_operation_impl<so_t, se_t> so_apply_se_perm_t;

    try {

    symmetry_element_set<2, double> set1(se_t::k_sym_type);
    symmetry_element_set<2, double> set2(se_t::k_sym_type);

    permutation<2> p0;
    scalar_transf<double> tr0, tr1(-1.0);
    params_t params(set1, p0, is_asym ? tr0 : tr1,
            sign ? tr0 : tr1, keep_zero, set2);

    so_apply_se_perm_t op;
    op.perform(params);
    if(!set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "!set2.is_empty() (1).");
    }

    permutation<2> perm; perm.permute(0, 1);
    se_t elem(perm, tr0);
    set2.insert(elem);

    op.perform(params);
    if(!set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
            "!set2.is_empty() (2).");
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests the application on a non-empty set
 **/
void so_apply_se_perm_test::test_2(bool keep_zero,
        bool is_asym, bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_apply_se_perm_test::test_2(" << keep_zero << ", "
            << is_asym << ", " << sign << ")";

    typedef se_perm<2, double> se_t;
    typedef so_apply<2, double> so_t;
    typedef symmetry_operation_params<so_t> params_t;
    typedef symmetry_operation_impl<so_t, se_t> so_apply_se_perm_t;

    try {

    permutation<2> p; p.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.0);
    se_t elem1(p, tr0);

    symmetry_element_set<2, double> set1(se_t::k_sym_type);
    symmetry_element_set<2, double> set2(se_t::k_sym_type);

    set1.insert(elem1);

    permutation<2> p0;
    params_t params(set1, p0, is_asym ? tr0 : tr1,
            sign ? tr0 : tr1, keep_zero, set2);

    so_apply_se_perm_t op;
    op.perform(params);

    if(set2.is_empty())
        fail_test(tnss.str().c_str(),
                __FILE__, __LINE__, "set2.is_empty()");

    symmetry_element_set_adapter<2, double, se_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se_t>::iterator i =
            adapter.begin();
    const se_t &elem2 = adapter.get_elem(i);
    i++;
    if(i != adapter.end())
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "Expected only one element.");

    if (elem2.get_transf() != tr0)
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "elem2.get_transf() != tr0");

    if(!elem1.get_perm().equals(elem2.get_perm()))
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "elem1 != elem2");

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Tests the application on a non-empty set with permutation
 **/
void so_apply_se_perm_test::test_3(bool keep_zero,
        bool is_asym, bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_apply_se_perm_test::test_3(" << keep_zero << ", "
            << is_asym << ", " << sign << ")";

    typedef se_perm<4, double> se_t;
    typedef so_apply<4, double> so_t;
    typedef symmetry_operation_params<so_t> params_t;
    typedef symmetry_operation_impl<so_t, se_t> so_apply_se_perm_t;

    try {

    scalar_transf<double> tr0, tr1(-1.);
    se_t el1(permutation<4>().permute(0, 1), tr1);
    se_t el2(permutation<4>().permute(2, 3), tr0);

    symmetry_element_set<4, double> set1(se_t::k_sym_type);
    symmetry_element_set<4, double> set2(se_t::k_sym_type);

    set1.insert(el1);
    set1.insert(el2);

    permutation<4> perm; perm.permute(0, 1).permute(1, 2);
    params_t params(set1, perm,
            is_asym ? tr0 : tr1, sign ? tr0 : tr1, keep_zero, set2);

    so_apply_se_perm_t op;
    op.perform(params);

    if(set2.is_empty())
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "set2.is_empty()");

    symmetry_element_set_adapter<4, double, se_t> adapter(set2);
    symmetry_element_set_adapter<4, double, se_t>::iterator i =
            adapter.begin();
    const se_t &elem1 = adapter.get_elem(i); i++;
    if (is_asym) {
        if(i != adapter.end())
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected one element.");

        if (elem1.get_transf() != tr0)
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "elem1.get_transf() != tr0");

        if (! elem1.get_perm().equals(permutation<4>().permute(1, 3))) {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Unexpected permutation in elem1.");
        }
    }
    else {
        if(i == adapter.end())
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "Expected two elements.");

        const se_t &elem2 = adapter.get_elem(i); i++;
        if(i != adapter.end())
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Expected only two elements.");

        bool is_p02;
        if (elem1.get_perm().equals(permutation<4>().permute(0, 2))) {
            is_p02 = true;
            if (sign) {
                if (elem1.get_transf() != tr0)
                    fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                            "elem1.get_transf() != tr0");
            }
            else {
                if (elem1.get_transf() != tr1)
                    fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                            "elem1.get_transf() != tr1");
            }
        }
        else if (elem1.get_perm().equals(permutation<4>().permute(1, 3))) {
            is_p02 = false;
            if (elem1.get_transf() != tr0)
                fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                        "elem1.get_transf() != tr0");
        }
        else {
            fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                    "Unexpected permutation in elem1.");
        }

        if (is_p02) {
            if (! elem2.get_perm().equals(permutation<4>().permute(1, 3)))
                fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                        "Unexpected permutation elem2.");

            if (elem2.get_transf() != tr0)
                fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                        "elem2.get_transf() !=  tr0");
        }
        else {
            if (! elem2.get_perm().equals(permutation<4>().permute(0, 2)))
                fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                        "Unexpected permutation elem2.");

            if (sign) {
                if (elem2.get_transf() != tr0)
                    fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                            "elem2.get_transf() != tr0");
            }
            else {
                if(elem2.get_transf() != tr1)
                    fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                            "elem2.get_transf() != tr1");
            }
        }
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
