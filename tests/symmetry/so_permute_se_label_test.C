#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_permute_se_label.h>
#include "../compare_ref.h"
#include "so_permute_se_label_test.h"

namespace libtensor {

void so_permute_se_label_test::perform() throw(libtest::test_exception) {

    std::string s6 = "S6";
    setup_pg_table(s6);

    try {

    test_1(s6);

    } catch (libtest::test_exception &e) {
        clear_pg_table(s6);
        throw;
    }

    clear_pg_table(s6);

}


/** \test Permutes a group with one element of Au symmetry.
 **/
void so_permute_se_label_test::test_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_permute_se_label_test::test_1(" << table_id << ")";
    std::string tns = tnss.str();

    typedef se_label<4, double> se4_t;
    typedef so_permute<4, double> so_permute_t;
    typedef symmetry_operation_impl<so_permute_t, se4_t>
        so_permute_se_t;

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 8; i2[3] = 8;

    block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

    mask<4> m, ma, mb, mc, md;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    ma[0] = true; ma[1] = true; mb[2] = true; mb[3] = true;
    mc[0] = true; md[1] = true; mc[2] = true; md[3] = true;
    bis.split(m, 2); bis.split(m, 4); bis.split(m, 6);

    se4_t el(bis.get_block_index_dims(), table_id);
    {
        block_labeling<4> &bl = el.get_labeling();
        for (unsigned int i = 0; i < 4; i++) {
            bl.assign(ma, i, i);
        }

        bl.assign(mb, 0, 3);
        bl.assign(mb, 1, 0);
        bl.assign(mb, 2, 1);
        bl.assign(mb, 3, 2);
        el.set_rule(2);
    }
    permutation<4> perm;
    perm.permute(0, 1).permute(1, 2);
    bis.permute(perm);

    se4_t el_ref(bis.get_block_index_dims(), table_id);
    {
        block_labeling<4> &bl_ref = el_ref.get_labeling();
        for (unsigned int i = 0; i < 4; i++) {
            bl_ref.assign(mc, i, i);
        }
        bl_ref.assign(md, 0, 3); bl_ref.assign(md, 1, 0);
        bl_ref.assign(md, 2, 1); bl_ref.assign(md, 3, 2);

        el_ref.set_rule(2);
    }

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

    set1.insert(el);
    set2_ref.insert(el_ref);

    symmetry_operation_params<so_permute_t> params(set1, perm, set2);

    so_permute_se_t().perform(params);

    compare_ref<4>::compare(tns.c_str(), bis, set2, set2_ref);

    if(set2.is_empty()) {
        fail_test(tns.c_str(), __FILE__, __LINE__, "Expected a non-empty set.");
    }

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}






} // namespace libtensor
