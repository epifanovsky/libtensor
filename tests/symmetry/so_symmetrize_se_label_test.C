#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_symmetrize_se_label.h>
#include "so_symmetrize_se_label_test.h"

namespace libtensor {

void so_symmetrize_se_label_test::perform() throw(libtest::test_exception) {

    std::string table_id = "S6";
    setup_pg_table(table_id);

    try {

    test_empty(table_id);
    test_sym2_1(table_id);
    test_sym2_2(table_id);
    test_sym2_3(table_id);
    test_sym3_1(table_id);

    } catch (libtest::test_exception &e) {
        clear_pg_table(table_id);
        throw;
    }

    clear_pg_table(table_id);
}


/** \test Tests that a symmetrization of an empty group yields an
        empty group
 **/
void so_symmetrize_se_label_test::test_empty(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_label_test::test_empty(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);

    sequence<4, size_t> idxgrp(0), symidx(0);
    idxgrp[0] = 1; idxgrp[1] = 2; idxgrp[2] = 3;
    symidx[0] = 1; symidx[1] = 1; symidx[2] = 1;

    scalar_transf<double> trp(1.), trc(1.0);
    symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
            symidx, trp, trc, set2);

    so_symmetrize_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__, "Expected an empty set.");
    }

    } catch (exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Symmetrization of 2 dim of a 4-space.
 **/
void so_symmetrize_se_label_test::test_sym2_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_label_test::test_sym2_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

    libtensor::index<4> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3;
    dimensions<4> bidims1(index_range<4>(i1a, i1b));
    se4_t el1(bidims1, table_id);
    {
    block_labeling<4> &bl1 = el1.get_labeling();
    mask<4> m1; m1[0] = m1[1] = m1[2] = m1[3] = true;
    for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);

    sequence<4, size_t> s1(0), s2(0);
    s1[0] = s1[1] = s1[2] = 1; s2[3] = 1;
    evaluation_rule<4> r1;
    product_rule<4> &pr1a = r1.new_product();
    pr1a.add(s1, 0); pr1a.add(s2, 0);
    product_rule<4> &pr1b = r1.new_product();
    pr1b.add(s1, 1); pr1b.add(s2, 1);
    product_rule<4> &pr1c = r1.new_product();
    pr1c.add(s1, 2); pr1c.add(s2, 2);
    product_rule<4> &pr1d = r1.new_product();
    pr1d.add(s1, 3); pr1d.add(s2, 3);
    el1.set_rule(r1);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);

    set1.insert(el1);

    sequence<4, size_t> idxgrp(0), symidx(0);
    idxgrp[2] = 1; idxgrp[3] = 2;
    symidx[2] = 1; symidx[3] = 1;

    scalar_transf<double> trp, trc;
    symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
            symidx, trp, trc, set2);

    so_symmetrize_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
    symmetry_element_set_adapter<4, double, se4_t>::iterator it =
            adapter.begin();
    const se4_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<4> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);

    abs_index<4> ai(bidims2);
    do {

        if (el1.is_allowed(ai.get_index())) {
            libtensor::index<4> idx(ai.get_index());
            std::swap(idx[2], idx[3]);

            rx[ai.get_abs_index()] = true;
            rx[abs_index<4>(idx, bidims2).get_abs_index()] = true;
        }
    } while (ai.inc());

    check_allowed(tns.c_str(), "el2", el2, rx);

    } catch (exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Symmetrization of 2 dim of a 4-space.
 **/
void so_symmetrize_se_label_test::test_sym2_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_label_test::test_sym2_2(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

    libtensor::index<4> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3;
    dimensions<4> bidims1(index_range<4>(i1a, i1b));
    se4_t el1(bidims1, table_id);
    {
    block_labeling<4> &bl1 = el1.get_labeling();
    mask<4> m1; m1[0] = m1[1] = m1[2] = m1[3] = true;
    for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);

    sequence<4, size_t> s1(0), s2(0);
    s1[0] = s1[1] = 1; s2[2] = s2[3] = 1;
    evaluation_rule<4> r1;
    product_rule<4> &pr1 = r1.new_product();
    pr1.add(s1, 1); pr1.add(s2, 2);
    el1.set_rule(r1);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    set1.insert(el1);

    sequence<4, size_t> idxgrp(0), symidx(0);
    idxgrp[0] = idxgrp[1] = 1; idxgrp[2] = idxgrp[3] = 2;
    symidx[0] = symidx[2] = 1; symidx[1] = symidx[3] = 2;

    scalar_transf<double> trp, trc;
    symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
            symidx, trp, trc, set2);

    so_symmetrize_se_t().perform(params);
    if(set2.is_empty()) {
        fail_test(tns.c_str(),
                __FILE__, __LINE__, "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<4, double, se4_t> adapter(set2);
    symmetry_element_set_adapter<4, double, se4_t>::iterator it =
            adapter.begin();
    const se4_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<4> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);

    abs_index<4> ai(bidims2);
    do {
        const libtensor::index<4> &idx = ai.get_index();
        bool xij = (idx[0] == 0 && idx[1] == 1) ||
                (idx[0] == 1 && idx[1] == 0) ||
                (idx[0] == 1 && idx[1] == 1) ||
                (idx[0] == 2 && idx[1] == 3) ||
                (idx[0] == 3 && idx[1] == 2) ||
                (idx[0] == 3 && idx[1] == 3);
        bool ykl = (idx[2] == 0 && idx[3] == 2) ||
                (idx[2] == 1 && idx[3] == 3) ||
                (idx[2] == 2 && idx[3] == 0) ||
                (idx[2] == 3 && idx[3] == 1);
        bool yij = (idx[0] == 0 && idx[1] == 2) ||
                (idx[0] == 1 && idx[1] == 3) ||
                (idx[0] == 2 && idx[1] == 0) ||
                (idx[0] == 3 && idx[1] == 1);
        bool xkl = (idx[2] == 0 && idx[3] == 1) ||
                (idx[2] == 1 && idx[3] == 0) ||
                (idx[2] == 1 && idx[3] == 1) ||
                (idx[2] == 2 && idx[3] == 3) ||
                (idx[2] == 3 && idx[3] == 2) ||
                (idx[2] == 3 && idx[3] == 3);

        rx[ai.get_abs_index()] = ((xij && ykl) || (yij && xkl));

    } while (ai.inc());

    check_allowed(tns.c_str(), "el2", el2, rx);

    } catch (exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Double symmetrization of 2 dim of a 4-space.
 **/
void so_symmetrize_se_label_test::test_sym2_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_label_test::test_sym2_3(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef so_symmetrize<4, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se4_t> so_symmetrize_se_t;

    try {

    libtensor::index<4> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3;
    dimensions<4> bidims1(index_range<4>(i1a, i1b));
    se4_t el1(bidims1, table_id);
    {
    block_labeling<4> &bl1 = el1.get_labeling();
    mask<4> m1; m1[0] = m1[1] = m1[2] = m1[3] = true;
    for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);

    sequence<4, size_t> s1(0), s2(0);
    s1[0] = s1[2] = 1; s2[1] = s2[3] = 1;
    evaluation_rule<4> r1;
    product_rule<4> &pr1 = r1.new_product();
    pr1.add(s1, 0); pr1.add(s2, 0);
    el1.set_rule(r1);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    symmetry_element_set<4, double> set3(se4_t::k_sym_type);
    set1.insert(el1);

    sequence<4, size_t> ig1(0), ig2(0), si1(0), si2(0);
    ig1[0] = 1; ig1[1] = 2; ig2[2] = 1; ig2[3] = 2;
    si1[0] = si1[1] = 1; si2[2] = si2[3] = 1;

    scalar_transf<double> trp(-1.0), trc;
    symmetry_operation_params<so_symmetrize_t> params1(set1, ig1,
            si1, trp, trc, set2);
    so_symmetrize_se_t().perform(params1);
    symmetry_operation_params<so_symmetrize_t> params2(set2, ig2,
            si2, trp, trc, set3);
    so_symmetrize_se_t().perform(params2);

    if(set3.is_empty()) {
        fail_test(tns.c_str(),
                __FILE__, __LINE__, "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<4, double, se4_t> ad2(set2), ad3(set3);
    symmetry_element_set_adapter<4, double, se4_t>::iterator it2 =
            ad2.begin(), it3 = ad3.begin();
    const se4_t &el2 = ad2.get_elem(it2), &el3 = ad3.get_elem(it3);
    it2++; it3++;
    if(it2 != ad2.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element (2).");
    }
    if(it3 != ad3.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element (3).");
    }

    const dimensions<4> &bidims = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx2(bidims.get_size(), false),
            rx3(bidims.get_size(), false);

    abs_index<4> ai(bidims);
    do {
        const libtensor::index<4> &idx = ai.get_index();
        if (el1.is_allowed(idx)) {
            rx2[ai.get_abs_index()] = rx3[ai.get_abs_index()] = true;
            libtensor::index<4> idx2(idx);
            std::swap(idx2[0], idx2[1]);
            rx2[abs_index<4>(idx2, bidims).get_abs_index()] = true;
            rx3[abs_index<4>(idx2, bidims).get_abs_index()] = true;
            std::swap(idx2[2], idx2[3]);
            rx3[abs_index<4>(idx2, bidims).get_abs_index()] = true;
            std::swap(idx2[0], idx2[1]);
            rx3[abs_index<4>(idx2, bidims).get_abs_index()] = true;
        }

    } while (ai.inc());

    check_allowed(tns.c_str(), "el2", el2, rx2);
    check_allowed(tns.c_str(), "el3", el3, rx3);

    } catch (exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Symmetrization of 3 dim of a 3-space.
 **/
void so_symmetrize_se_label_test::test_sym3_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_symmetrize_se_label_test::test_sym3_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<3, double> se3_t;
    typedef so_symmetrize<3, double> so_symmetrize_t;
    typedef symmetry_operation_impl<so_symmetrize_t, se3_t> so_symmetrize_se_t;

    try {

    libtensor::index<3> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3;
    dimensions<3> bidims1(index_range<3>(i1a, i1b));

    se3_t el1(bidims1, table_id);
    {
    block_labeling<3> &bl1 = el1.get_labeling();
    mask<3> m1; m1[0] = m1[1] = m1[2] = true;
    for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);
    evaluation_rule<3> r1;
    sequence<3, size_t> seq1a(0), seq1b(0);
    seq1a[0] = seq1a[1] = 1;
    seq1b[1] = seq1b[2] = 1;
    product_rule<3> &pr1 = r1.new_product();
    pr1.add(seq1a, 0); pr1.add(seq1b, 0);
    el1.set_rule(r1);
    }

    symmetry_element_set<3, double> set1(se3_t::k_sym_type);
    symmetry_element_set<3, double> set2(se3_t::k_sym_type);
    set1.insert(el1);

    sequence<3, size_t> idxgrp(0), symidx(0);
    idxgrp[0] = 1; idxgrp[1] = 2; idxgrp[2] = 3;
    symidx[0] = symidx[1] = symidx[2] = 1;

    scalar_transf<double> trp, trc;
    symmetry_operation_params<so_symmetrize_t> params(set1, idxgrp,
            symidx, trp, trc, set2);

    so_symmetrize_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<3, double, se3_t> adapter(set2);
    symmetry_element_set_adapter<3, double, se3_t>::iterator it =
            adapter.begin();
    const se3_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<3> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);
    abs_index<3> ai(bidims2);
    do {
        const libtensor::index<3> &idx = ai.get_index();
        bool x011 = (idx[1] == idx[2]);
        bool x101 = (idx[0] == idx[2]);
        bool x110 = (idx[0] == idx[1]);

        rx[ai.get_abs_index()] =
                ((x110 && x011) || (x101 && x011) || (x110 && x101));

    } while (ai.inc());

    check_allowed(tns.c_str(), "el2", el2, rx);

    } catch (exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
