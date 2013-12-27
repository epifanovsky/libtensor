/* * sparse_loop_list_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparsity_fuser.h>
#include "sparsity_fuser_test.h"

using namespace std;

namespace libtensor {

namespace {

//C_(ij)k = A_(jil) B_lk
class contract_test_f {
public:
    std::vector< sparse_bispace_any_order > bispaces; 
    std::vector< block_loop > loops;

    contract_test_f() 
    {
        //Set up bispaces
        sparse_bispace<1> spb_i(2);
        vector<size_t> split_points_i;
        split_points_i.push_back(1);
        spb_i.split(split_points_i);

        sparse_bispace<1> spb_j(2);
        vector<size_t> split_points_j;
        split_points_j.push_back(1);
        spb_j.split(split_points_j);

        sparse_bispace<1> spb_k(2);
        vector<size_t> split_points_k;
        split_points_k.push_back(1);
        spb_k.split(split_points_k);

        sparse_bispace<1> spb_l(2);
        vector<size_t> split_points_l;
        split_points_l.push_back(1);
        spb_l.split(split_points_l);

        sparse_bispace<3> spb_C = spb_i % spb_j << std::vector< sequence<2,size_t> >() | spb_k;
        sparse_bispace<3> spb_A = spb_j % spb_i % spb_l << std::vector< sequence<3,size_t> >();
        sparse_bispace<2> spb_B = spb_l | spb_k;

        bispaces.push_back(spb_C);
        bispaces.push_back(spb_A);
        bispaces.push_back(spb_B);

        //Set up loop list
        loops.resize(4,block_loop(bispaces));
        //i loop
        loops[0].set_subspace_looped(0,0);
        loops[0].set_subspace_looped(1,1);

        //j loop
        loops[1].set_subspace_looped(0,1);
        loops[1].set_subspace_looped(1,0);

        //k loop
        loops[2].set_subspace_looped(0,2);
        loops[2].set_subspace_looped(2,1);

        //l loop
        loops[3].set_subspace_looped(1,2);
        loops[3].set_subspace_looped(2,0);
    }
};

} // namespace unnamed

void sparsity_fuser_test::perform() throw(libtest::test_exception) {
    test_get_loops_accessing_tree();
}

void sparsity_fuser_test::test_get_loops_accessing_tree() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_get_loops_accessing_tree()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    idx_list loop_indices_C = sf.get_loops_accessing_tree(0);
    size_t arr_C[2] = {0,1}; 
    idx_list correct_li_C(&arr_C[0],&arr_C[0]+2);

    if(loop_indices_C != correct_li_C)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_accessing_tree(...) returned incorrect value for C sparse tree");
    }
}

} // namespace libtensor
