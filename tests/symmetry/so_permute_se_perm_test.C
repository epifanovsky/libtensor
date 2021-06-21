#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_permute_se_perm.h>
#include "../compare_ref.h"
#include "so_permute_se_perm_test.h"

namespace libtensor {

void so_permute_se_perm_test::perform() {

    test_1();

}


/** \test Permutes a group with one element of Au symmetry.
 **/
void so_permute_se_perm_test::test_1() {

    static const char *testname = "so_permute_se_perm_test::test_1()";

    typedef se_perm<4, double> se4_t;
    typedef so_permute<4, double> so_permute_t;
    typedef symmetry_operation_impl<so_permute_t, se4_t>
        so_permute_se_t;

    try {

    libtensor::index<4> i4a, i4b;
    i4b[0] = 8; i4b[1] = 8; i4b[2] = 8; i4b[3] = 8;

    block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));

    mask<4> m4, m4a, m4b, m4c, m4d;
    m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
    bis4.split(m4, 2); bis4.split(m4, 4); bis4.split(m4, 6);


    permutation<4> perm;
    perm.permute(0, 1).permute(1, 2);
    bis4.permute(perm);

    symmetry_element_set<4, double> set1(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2(se4_t::k_sym_type);
    symmetry_element_set<4, double> set2_ref(se4_t::k_sym_type);

    scalar_transf<double> tr0, tr1(-1.0);
    set1.insert(se4_t(permutation<4>().permute(0, 1), tr1));
    set1.insert(se4_t(permutation<4>().permute(2, 3), tr1));
    set1.insert(se4_t(permutation<4>().permute(0, 2).permute(1, 3), tr1));
    set2_ref.insert(se4_t(permutation<4>().permute(2, 0), tr1));
    set2_ref.insert(se4_t(permutation<4>().permute(1, 3), tr1));
    set2_ref.insert(se4_t(permutation<4>().permute(2, 1).permute(0, 3), tr0));

    symmetry_operation_params<so_permute_t> params(set1, perm, set2);

    so_permute_se_t().perform(params);

    if(set2.is_empty()) {
        fail_test(testname, __FILE__, __LINE__, "Expected a non-empty set.");
    }

    compare_ref<4>::compare(testname, bis4, set2, set2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}






} // namespace libtensor
