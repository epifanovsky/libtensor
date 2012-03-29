#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_apply_se_label.h>
#include "../compare_ref.h"
#include "so_apply_se_label_test.h"

namespace libtensor {

void so_apply_se_label_test::perform() throw(libtest::test_exception) {

    std::string table_id = setup_pg_table();

    try {

    test_1(table_id, false,  true, false);
    test_1(table_id, false, false, false);
    test_1(table_id, false, false,  true);
    test_1(table_id,  true,  true, false);
    test_1(table_id,  true, false, false);
    test_1(table_id,  true, false,  true);

    } catch (std::exception &e) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }

    product_table_container::get_instance().erase(table_id);

}


/** \test Tests application on a group with permutation
 **/
void so_apply_se_label_test::test_1(
        const std::string &table_id, bool keep_zero,
        bool is_asym, bool sign) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_apply_se_label_test::test_1(" << table_id << ", "
            << keep_zero << ", " << is_asym << ", " << sign << ")";

    typedef se_label<2, double> se2_t;
    typedef so_apply<2, double> so_t;
    typedef symmetry_operation_impl<so_t, se2_t> so_se_t;

    index<2> i1, i2;
    i2[0] = i2[1] = 3;

    dimensions<2> bidims(index_range<2>(i1, i2));

    mask<2> m;
    m[0] = m[1] = true;

    se2_t el1(bidims, table_id);
    block_labeling<2> &bl1 = el1.get_labeling();
    for (unsigned int i = 0; i < 4; i++) {
        bl1.assign(m, i, i);
    }

    el1.set_rule(2);

    permutation<2> p;
    scalar_transf<double> tr0, tr1(-1.0);

    symmetry_element_set<2, double> set1(se2_t::k_sym_type);
    symmetry_element_set<2, double> set2(se2_t::k_sym_type);

    set1.insert(el1);
    symmetry_operation_params<so_t> params(set1, p,
            is_asym ? tr0 : tr1, sign ? tr0 : tr1, keep_zero, set2);

    so_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "Expected a non-empty set.");
    }

    symmetry_element_set_adapter<2, double, se2_t> adc(set2);
    symmetry_element_set_adapter<2, double, se2_t>::iterator it =
            adc.begin();
    const se2_t &el2 = adc.get_elem(it);
    it++;
    if (it != adc.end()) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__,
                "More than 1 element in set.");
    }

    std::vector<bool> rx(bidims.get_size(), ! keep_zero);
    if (keep_zero) {
        rx[2] = rx[7] = rx[8] = rx[13] = true;
    }

    check_allowed(tnss.str().c_str(), "el2", el2, rx);
}






} // namespace libtensor
