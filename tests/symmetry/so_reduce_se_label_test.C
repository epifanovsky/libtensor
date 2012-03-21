#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_reduce_se_label.h>
#include "so_reduce_se_label_test.h"

namespace libtensor {

void so_reduce_se_label_test::perform() throw(libtest::test_exception) {

    std::string table_id = setup_pg_table();

    try {

        test_empty_1(table_id);
        test_empty_2(table_id);
        test_nm1_1(table_id);
        test_nm1_2(table_id, true);
        test_nm1_2(table_id, false);
        test_nmk_1(table_id);
        test_nmk_2(table_id, true);
        test_nmk_2(table_id, false);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }

    product_table_container::get_instance().erase(table_id);
}


/**	\test Tests that a single reduction of 2 dim of an empty group yields an
        empty group of lower order
 **/
void so_reduce_se_label_test::test_empty_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_empty_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef se_label<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    mask<4> msk; msk[2] = msk[3] = true;
    sequence<4, size_t> seq(0);
    index<4> ia, ib; ib[2] = ib[3] = 2;
    index_range<4> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__, "Expected an empty set.");
    }
}

/** \test Tests that a double reduction of dimensions of an empty group
        yields an empty group of lower order
 **/
void so_reduce_se_label_test::test_empty_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_empty_2(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<5, double> se5_t;
    typedef se_label<1, double> se1_t;
    typedef so_reduce<5, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se1_t> so_reduce_se_t;

    symmetry_element_set<5, double> set1(se5_t::k_sym_type);
    symmetry_element_set<1, double> set2(se1_t::k_sym_type);

    mask<5> msk; msk[0] = msk[1] = msk[2] = msk[3] = true;
    sequence<5, size_t> seq(0); seq[2] = seq[3] = 1;
    index<5> ia, ib; ib[0] = ib[1] = ib[2] = ib[3] = 2;
    index_range<5> ir(ia, ib);
    symmetry_operation_params<so_reduce_t> params(set1, msk, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(!set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__, "Expected an empty set.");
    }

}

/**	\test Reduction of 2 dim of a 3-space on a 1-space in one step.
 **/
void so_reduce_se_label_test::test_nm1_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_nm1_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<1, double> se1_t;
    typedef se_label<3, double> se3_t;
    typedef so_reduce<3, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se1_t> so_reduce_se_t;

    index<3> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3;
    dimensions<3> bidims1(index_range<3>(i1a, i1b));
    se3_t el1(bidims1, table_id);
    {
        block_labeling<3> &bl1 = el1.get_labeling();
        mask<3> m1; m1[0] = m1[1] = m1[2] = true;
        for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);
        el1.set_rule(2);
    }

    symmetry_element_set<3, double> set1(se3_t::k_sym_type);
    symmetry_element_set<1, double> set2(se1_t::k_sym_type);

    set1.insert(el1);
    mask<3> m; m[1] = m[2] = true;
    sequence<3, size_t> seq;
    index_range<3> ir(i1a, i1b);
    symmetry_operation_params<so_reduce_t> params(set1, m, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<1, double, se1_t> adapter(set2);
    symmetry_element_set_adapter<1, double, se1_t>::iterator it =
            adapter.begin();
    const se1_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<1> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);

    rx[2] = rx[3] = true;

    check_allowed(tns.c_str(), "el2", el2, rx);
}


/**	\test Single reduction of 2 dim of a 4-space on a 2-space.
 **/
void so_reduce_se_label_test::test_nm1_2(const std::string &table_id,
        bool product) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_nm1_2(" <<
            table_id << ", " << product << ")";
    std::string tns(tnss.str());

    typedef se_label<4, double> se4_t;
    typedef se_label<2, double> se2_t;
    typedef so_reduce<4, 2, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    index<4> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3;
    dimensions<4> bidims1(index_range<4>(i1a, i1b));

    se4_t el1(bidims1, table_id);
    {
        block_labeling<4> &bl1 = el1.get_labeling();
        mask<4> m1; m1[0] = m1[1] = m1[2] = m1[3] = true;
        for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);
        evaluation_rule<4> r1;
        sequence<4, size_t> seq1a(0), seq1b(0);
        seq1a[0] = seq1a[1] = 1;
        seq1b[2] = seq1b[3] = 1;
        r1.add_sequence(seq1a);
        r1.add_sequence(seq1b);
        r1.add_product(0, 2, 0);
        if (product)
            r1.add_to_product(0, 1, 0, 0);
        else
            r1.add_product(1, 0, 0);
        el1.set_rule(r1);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);
    mask<4> m; m[1] = m[2] = true;
    sequence<4, size_t> seq(0);
    index_range<4> ir(i1a, i1b);
    symmetry_operation_params<so_reduce_t> params(set1, m, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator it =
            adapter.begin();
    const se2_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<2> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);
    if (product) {
        rx[2] = rx[7] = rx[8] = rx[13] = true;
    }
    else {
        rx.assign(bidims2.get_size(), true);
    }
    check_allowed(tns.c_str(), "el2", el2, rx);
}

/** \test Double reduction of 4 dim of a 6-space on a 2-space (simple rule)
 **/
void so_reduce_se_label_test::test_nmk_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_nmk_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<6, double> se6_t;
    typedef se_label<2, double> se2_t;
    typedef so_reduce<6, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    index<6> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3; i1b[4] = 3; i1b[5] = 3;
    dimensions<6> bidims1(index_range<6>(i1a, i1b));

    se6_t el1(bidims1, table_id);
    {
        block_labeling<6> &bl1 = el1.get_labeling();
        mask<6> m1;
        m1[0] = m1[1] = m1[2] = m1[3] = m1[4] = m1[5] = true;
        for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);
        el1.set_rule(0);
    }

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);
    set1.insert(el1);

    mask<6> m; m[1] = m[2] = m[3] = m[4] = true;
    sequence<6, size_t> seq(0); seq[2] = seq[4] = 1;
    index_range<6> ir(i1a, i1b);
    symmetry_operation_params<so_reduce_t> params(set1, m, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator it =
            adapter.begin();
    const se2_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<2> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);
    rx[0] = rx[1] = rx[4] = rx[5] =
            rx[10] = rx[11] = rx[14] = rx[15] = true;

    check_allowed(tns.c_str(), "el2", el2, rx);
}

/** \test Double reduction of 4 dim of a 6-space on a 2-space (complex rule)
 **/
void so_reduce_se_label_test::test_nmk_2(const std::string &table_id,
        bool product) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_reduce_se_label_test::test_nmk_2("
            << table_id << "," << product << ")";
    std::string tns(tnss.str());

    typedef se_label<6, double> se6_t;
    typedef se_label<2, double> se2_t;
    typedef so_reduce<6, 4, double> so_reduce_t;
    typedef symmetry_operation_impl<so_reduce_t, se2_t> so_reduce_se_t;

    index<6> i1a, i1b;
    i1b[0] = 3; i1b[1] = 3; i1b[2] = 3; i1b[3] = 3; i1b[4] = 3; i1b[5] = 3;
    dimensions<6> bidims1(index_range<6>(i1a, i1b));

    se6_t el1(bidims1, table_id);
    {
        block_labeling<6> &bl1 = el1.get_labeling();
        mask<6> m1;
        m1[0] = m1[1] = m1[2] = m1[3] = m1[4] = m1[5] = true;
        for (size_t i = 0; i < 4; i++) bl1.assign(m1, i, i);
        evaluation_rule<6> r1;
        sequence<6, size_t> seq1a(0), seq1b(0);
        seq1a[0] = seq1a[1] = seq1a[2] = 1;
        seq1b[3] = seq1b[4] = seq1b[5] = 1;
        r1.add_sequence(seq1a);
        r1.add_sequence(seq1b);
        r1.add_product(0, 2, 0);
        if (product)
            r1.add_to_product(0, 1, 2, 0);
        else
            r1.add_product(1, 2, 0);
        el1.set_rule(r1);
    }

    symmetry_element_set<6, double> set1(se6_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);
    mask<6> m; m[1] = m[2] = m[4] = m[5] = true;
    sequence<6, size_t> seq(0); seq[2] = seq[5] = 1;
    index_range<6> ir(i1a, i1b);
    symmetry_operation_params<so_reduce_t> params(set1, m, seq, ir, ir, set2);

    so_reduce_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<2, double, se2_t> adapter(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator it =
            adapter.begin();
    const se2_t &el2 = adapter.get_elem(it);
    it++;
    if(it != adapter.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    const dimensions<2> &bidims2 = el2.get_labeling().get_block_index_dims();
    std::vector<bool> rx(bidims2.get_size(), false);

    if (product) {
        rx[0] = rx[1] = rx[4] = rx[5] =
                rx[10] = rx[11] = rx[14] = rx[15] = true;
    }
    else {
        rx.assign(bidims2.get_size(), true);
    }
    check_allowed(tns.c_str(), "el2", el2, rx);
}

} // namespace libtensor
