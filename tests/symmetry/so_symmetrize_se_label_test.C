#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_symmetrize_se_label.h>
#include "so_symmetrize_se_label_test.h"

namespace libtensor {

void so_symmetrize_se_label_test::perform() throw(libtest::test_exception) {

    std::string table_id = setup_pg_table();

    try {

    test_empty(table_id);
    test_sym2_1(table_id);
    test_sym2_1(table_id);
    test_sym3_1(table_id);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }

    product_table_container::get_instance().erase(table_id);
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

    index<4> i1a, i1b;
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
    r1.add_sequence(s1);
    r1.add_sequence(s2);
    r1.add_product(0, 0, 0); r1.add_to_product(0, 1, 0, 0);
    r1.add_product(0, 1, 0); r1.add_to_product(1, 1, 1, 0);
    r1.add_product(0, 2, 0); r1.add_to_product(2, 1, 2, 0);
    r1.add_product(0, 3, 0); r1.add_to_product(3, 1, 3, 0);
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
            index<4> idx(ai.get_index());
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

    index<4> i1a, i1b;
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
    r1.add_sequence(s1);
    r1.add_sequence(s2);
    r1.add_product(0, 1, 0);
    r1.add_to_product(0, 1, 2, 0);
    el1.set_rule(r1);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    set1.insert(el1);

    sequence<4, size_t> idxgrp(0), symidx(0);
    idxgrp[0] = idxgrp[2] = 1; idxgrp[1] = idxgrp[3] = 2;
    symidx[0] = symidx[1] = 1; symidx[2] = symidx[3] = 2;

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
        const index<4> &idx = ai.get_index();
        bool xij = (idx[0] == 0 && idx[1] == 1) ||
                (idx[0] == 1 && idx[1] == 0) ||
                (idx[0] == 2 && idx[1] == 3) ||
                (idx[0] == 3 && idx[1] == 2) ||
                (idx[0] == 3 && idx[1] == 3);
        bool ykl = (idx[2] == 0 && idx[3] == 2) ||
                (idx[2] == 2 && idx[3] == 0) ||
                (idx[2] == 1 && idx[3] == 3) ||
                (idx[2] == 3 && idx[3] == 1);
        bool yij = (idx[0] == 0 && idx[1] == 2) ||
                (idx[0] == 2 && idx[1] == 0) ||
                (idx[0] == 1 && idx[1] == 3) ||
                (idx[0] == 3 && idx[1] == 1);
        bool xkl = (idx[2] == 0 && idx[3] == 1) ||
                (idx[2] == 1 && idx[3] == 0) ||
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

    index<3> i1a, i1b;
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
    r1.add_sequence(seq1a);
    r1.add_sequence(seq1b);
    r1.add_product(0, 0, 0);
    r1.add_to_product(0, 1, 0, 0);
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
        const index<3> &idx = ai.get_index();
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
