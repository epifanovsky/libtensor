#include <libtensor/symmetry/label/so_dirsum_impl_label.h>
#include "../compare_ref.h"
#include "so_dirsum_impl_label_test.h"

namespace libtensor {

void so_dirsum_impl_label_test::perform() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_impl_label_test::perform()";

    std::string table_id = setup_pg_table();

    try {

        test_empty_1(table_id);
        test_empty_2(table_id, true);
        test_empty_2(table_id, false);
        test_empty_3(table_id, true);
        test_empty_3(table_id, false);
        test_nn_1(table_id);
        test_nn_2(table_id);
        test_nn_3(table_id);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }

    product_table_container::get_instance().erase(table_id);
}


/**	\test Tests that the direct sum of two empty group yields an empty
        group of a higher order
 **/
void so_dirsum_impl_label_test::test_empty_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_empty_1(" << table_id << ")";
    std::string tns(tnss.str());

    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef se_label<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

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

    so_impl_t().perform(params);

    compare_ref<5>::compare(tns.c_str(), bisc, setc, setc_ref);

    if(! setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__, "Expected an empty set.");
    }
}


/**	\test Direct sum of a group with Au symmetry in 2-space and an empty
        group (1-space) forming a 3-space.
 **/
void so_dirsum_impl_label_test::test_empty_2(const std::string &table_id,
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_empty_2("
            << table_id << "," << perm << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<2, 1, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
    index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

    block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
    block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

    mask<2> ma; ma[0] = true; ma[1] = true;
    bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
    mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
    bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

    dimensions<2> bidimsa = bisa.get_block_index_dims();
    dimensions<3> bidimsc = bisc.get_block_index_dims();

    se2_t elema(bidimsa, table_id);
    { // Assign block labels
        block_labeling<2> &bla = elema.get_labeling();
        for (size_t i = 0; i < 4; i++) bla.assign(ma, i, i);
        elema.set_rule(2);
    }

    symmetry_element_set<2, double> seta(se2_t::k_sym_type);
    symmetry_element_set<1, double> setb(se1_t::k_sym_type);
    symmetry_element_set<3, double> setc(se3_t::k_sym_type);

    seta.insert(elema);

    permutation<3> px;
    if (perm) px.permute(0, 1).permute(1, 2);
    symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

    so_impl_t().perform(params);

    if(setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<3, double, se3_t> adc(setc);
    symmetry_element_set_adapter<3, double, se3_t>::iterator it =
            adc.begin();
    const se3_t &elemc = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    std::vector<bool> rx(bidimsc.get_size(), true);

    check_allowed(tns.c_str(), "elemc", elemc, rx);
}

/**	\test Direct sum of an empty group (1-space) and a group with Au symmetry
        in 2-space forming a 3-space.
 **/
void so_dirsum_impl_label_test::test_empty_3(const std::string &table_id,
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_empty_3("
            << table_id << "," << perm << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
    index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

    block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
    block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

    mask<2> mb; mb[0] = true; mb[1] = true;
    bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
    mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
    bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

    dimensions<2> bidimsb = bisb.get_block_index_dims();
    dimensions<3> bidimsc = bisc.get_block_index_dims();

    se2_t elemb(bidimsb, table_id);
    { // Assign block labels
        block_labeling<2> &blb = elemb.get_labeling();
        for (size_t i = 0; i < 4; i++) blb.assign(mb, i, i);
        elemb.set_rule(3);
    }

    symmetry_element_set<1, double> seta(se1_t::k_sym_type);
    symmetry_element_set<2, double> setb(se2_t::k_sym_type);
    symmetry_element_set<3, double> setc(se3_t::k_sym_type);

    setb.insert(elemb);

    permutation<3> px;
    if (perm) px.permute(0, 1).permute(1, 2);
    symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

    so_impl_t().perform(params);

    if(setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<3, double, se3_t> adc(setc);
    symmetry_element_set_adapter<3, double, se3_t>::iterator it =
            adc.begin();
    const se3_t &elemc = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    std::vector<bool> rx(bidimsc.get_size(), true);

    check_allowed(tns.c_str(), "elemc", elemc, rx);
}


/** \test Direct sum of a group with Ag symmetry in 1-space and a group
        with Eu symmetry in 2-space.
 **/
void so_dirsum_impl_label_test::test_nn_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_nn_1(" << table_id << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

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

    dimensions<1> bidimsa = bisa.get_block_index_dims();
    dimensions<2> bidimsb = bisb.get_block_index_dims();
    dimensions<3> bidimsc = bisc.get_block_index_dims();

    se1_t elema(bisa.get_block_index_dims(), table_id);
    {
        block_labeling<1> &bla = elema.get_labeling();
        for (size_t i = 0; i < 4; i++) bla.assign(ma, i, i);
        elema.set_rule(0);
    }

    se2_t elemb(bisb.get_block_index_dims(), table_id);
    {
        block_labeling<2> &blb = elemb.get_labeling();
        for (size_t i = 0; i < 4; i++) blb.assign(mb, i, i);
        elemb.set_rule(3);
    }

    symmetry_element_set<1, double> seta(se1_t::k_sym_type);
    symmetry_element_set<2, double> setb(se2_t::k_sym_type);
    symmetry_element_set<3, double> setc(se3_t::k_sym_type);

    seta.insert(elema);
    setb.insert(elemb);

    permutation<3> px;
    symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

    so_impl_t().perform(params);

    if(setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<3, double, se3_t> adc(setc);
    symmetry_element_set_adapter<3, double, se3_t>::iterator it =
            adc.begin();
    const se3_t &elemc = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    std::vector<bool> rx(bidimsc.get_size(), false);
    index<3> idx;
    size_t ii = 0, ij = 1, ik = 2;
    for (size_t i = 0; i < 4; i++) {
        idx[ii] = i;
        idx[ij] = 0; idx[ik] = 3;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 1; idx[ik] = 2;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 1; idx[ik] = 3;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 2; idx[ik] = 1;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 3; idx[ik] = 0;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 3; idx[ik] = 1;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
    }
    idx[ii] = 0;
    for (size_t j = 0; j < 4; j++) {
        idx[ij] = j;
        for (size_t k = 0; k < 4; k++) {
            idx[ik] = k;
            rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        }
    }

    check_allowed(tns.c_str(), "elemc", elemc, rx);
}

/** \test Direct sum of a group in 1-space with Eg symmetry and a group in
        2-space with Au symmetry. The result is permuted with [012->120].
 **/
void so_dirsum_impl_label_test::test_nn_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_nn_2(" << table_id << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

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

    dimensions<1> bidimsa = bisa.get_block_index_dims();
    dimensions<2> bidimsb = bisb.get_block_index_dims();
    dimensions<3> bidimsc = bisc.get_block_index_dims();

    se1_t elema(bisa.get_block_index_dims(), table_id);
    {
        block_labeling<1> &bla = elema.get_labeling();
        for (size_t i = 0; i < 4; i++) bla.assign(ma, i, i);
        elema.set_rule(1);
    }

    se2_t elemb(bisb.get_block_index_dims(), table_id);
    {
        block_labeling<2> &blb = elemb.get_labeling();
        for (size_t i = 0; i < 4; i++) blb.assign(mb, i, i);
        elemb.set_rule(2);
    }

    symmetry_element_set<1, double> seta(se1_t::k_sym_type);
    symmetry_element_set<2, double> setb(se2_t::k_sym_type);
    symmetry_element_set<3, double> setc(se3_t::k_sym_type);

    seta.insert(elema);
    setb.insert(elemb);

    permutation<3> px;
    px.permute(0, 1).permute(1, 2);
    symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

    so_impl_t().perform(params);

    if(setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<3, double, se3_t> adc(setc);
    symmetry_element_set_adapter<3, double, se3_t>::iterator it =
            adc.begin();
    const se3_t &elemc = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    std::vector<bool> rx(bidimsc.get_size(), false);
    size_t ii = 2, ij = 0, ik = 1;

    index<3> idx;
    for (size_t i = 0; i < 4; i++) {
        idx[ii] = i;
        idx[ij] = 0; idx[ik] = 2;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 1; idx[ik] = 3;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 2; idx[ik] = 0;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        idx[ij] = 3; idx[ik] = 1;
        rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
    }
    idx[ii] = 1;
    for (size_t j = 0; j < 4; j++) {
        idx[ij] = j;
        for (size_t k = 0; k < 4; k++) {
            idx[ik] = k;
            rx[abs_index<3>(idx, bidimsc).get_abs_index()] = true;
        }
    }

    check_allowed(tns.c_str(), "elemc", elemc, rx);
}

/** \test Direct sum of two groups in 2-space one with composite rule.
 **/
void so_dirsum_impl_label_test::test_nn_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_nn_3(" << table_id << ")";
    std::string tns = tnss.str();

    typedef se_label<2, double> se2_t;
    typedef se_label<4, double> se4_t;
    typedef so_dirsum<2, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se4_t> so_impl_t;

    index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
    index<4> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3;

    block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
    block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

    mask<2> ma; ma[0] = true; ma[1] = true;
    bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
    mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
    bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

    dimensions<2> bidimsa = bisa.get_block_index_dims();
    dimensions<4> bidimsc = bisc.get_block_index_dims();

    se2_t elema(bisa.get_block_index_dims(), table_id);
    {
        block_labeling<2> &bla = elema.get_labeling();
        for (size_t i = 0; i < 4; i++) bla.assign(ma, i, i);
        evaluation_rule<2> ra;
        basic_rule<2> bra1, bra2;
        bra1[0] = 1; bra1.set_target(1);
        bra2[1] = 1; bra2.set_target(2);
        evaluation_rule<2>::rule_id_t ira1 = ra.add_rule(bra1);
        evaluation_rule<2>::rule_id_t ira2 = ra.add_rule(bra2);
        ra.add_product(ira1);
        ra.add_to_product(0, ira2);
        elema.set_rule(ra);
    }

    se2_t elemb(bisa.get_block_index_dims(), table_id);
    {
        block_labeling<2> &blb = elemb.get_labeling();
        for (size_t i = 0; i < 4; i++) blb.assign(ma, i, i);
        elemb.set_rule(1);
    }

    symmetry_element_set<2, double> seta(se2_t::k_sym_type);
    symmetry_element_set<2, double> setb(se2_t::k_sym_type);
    symmetry_element_set<4, double> setc(se4_t::k_sym_type);

    seta.insert(elema);
    setb.insert(elemb);

    permutation<4> px;
    symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

    so_impl_t().perform(params);

    if(setc.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<4, double, se4_t> adc(setc);
    symmetry_element_set_adapter<4, double, se4_t>::iterator it =
            adc.begin();
    const se4_t &elemc = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Expected only one element.");
    }

    std::vector<bool> rx(bidimsc.get_size(), false);
    index<4> idx;
    size_t ii = 0, ij = 1, ik = 2, il = 3;
    idx[ii] = 1; idx[ij] = 2;
    for (size_t k = 0; k < 4; k++) {
        idx[ik] = k;
        for (size_t l = 0; l < 4; l++) {
            idx[il] = l;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
        }
    }

    for (size_t i = 0; i < 4; i++) {
        idx[ii] = i;
        for (size_t j = 0; j < 4; j++) {
            idx[ij] = j;
            idx[ik] = 0; idx[il] = 1;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
            idx[ik] = 1; idx[il] = 0;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
            idx[ik] = 1; idx[il] = 1;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
            idx[ik] = 2; idx[il] = 3;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
            idx[ik] = 3; idx[il] = 2;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
            idx[ik] = 3; idx[il] = 3;
            rx[abs_index<4>(idx, bidimsc).get_abs_index()] = true;
        }
    }

    check_allowed(tns.c_str(), "elemc", elemc, rx);
}


} // namespace libtensor
