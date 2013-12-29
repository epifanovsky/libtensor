/* * sparse_loop_list_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparsity_fuser.h>
#include "sparsity_fuser_test.h"

using namespace std;

namespace libtensor {

void sparsity_fuser_test::perform() throw(libtest::test_exception) {
    test_get_loops_for_tree();
    test_get_trees_for_loop();
    test_fuse();
}

//Test fixtures
namespace {

//C_(ij)k = A_(jil) B_lk
class contract_test_f {
public:
    vector< sparse_bispace_any_order > bispaces; 
    vector< block_loop > loops;

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

void sparsity_fuser_test::test_get_loops_for_tree() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_get_loops_for_tree()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    idx_list loop_indices_C = sf.get_loops_for_tree(0);
    size_t arr_C[2] = {0,1}; 
    idx_list correct_li_C(&arr_C[0],&arr_C[0]+2);

    if(loop_indices_C != correct_li_C)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_and_subspaces_for_tree(...) returned incorrect value for C sparse tree");
    }

    idx_list loop_indices_A = sf.get_loops_for_tree(1);
    size_t arr_A[3] = {0,1,3}; 
    idx_list correct_li_A(&arr_A[0],&arr_A[0]+3);

    if(loop_indices_A != correct_li_A)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_and_subspaces_for_tree(...) returned incorrect value for A sparse tree");
    }
}

void sparsity_fuser_test::test_get_trees_for_loop() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_get_trees_for_loop()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    idx_list trees_i = sf.get_trees_for_loop(0);
    size_t arr_i[2] = {0,1}; 
    idx_list correct_trees_i(&arr_i[0],&arr_i[0]+2);
    if(trees_i != correct_trees_i)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for i loop");
    }

    idx_list trees_j = sf.get_trees_for_loop(1);
    size_t arr_j[2] = {0,1}; 
    idx_list correct_trees_j(&arr_j[0],&arr_j[0]+2);
    if(trees_j != correct_trees_j)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for j loop");
    }

    idx_list trees_k = sf.get_trees_for_loop(2);
    idx_list correct_trees_k;
    if(trees_k != correct_trees_k)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for k loop");
    }

    idx_list trees_l = sf.get_trees_for_loop(3);
    size_t arr_l[1] = {1}; 
    idx_list correct_trees_l(&arr_l[0],&arr_l[0]+1);
    if(trees_l != correct_trees_l)
    {
        __asm__ __volatile__("int $3"); 
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for l loop");
    }
}

void sparsity_fuser_test::test_fuse() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_fuse()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    size_t fused_loops_arr[2] = {0,1};
    idx_list fused_loops(&fused_loops_arr[0],&fused_loops_arr[0]+2);
    sf.fuse(0,1,fused_loops);

    //The rhs tree should have disappeared from association with the relevant loops
    //The lhs tree should have been added to all loops with which the rhs tree was previously associated
    size_t arr_ijl[1] = {0}; 
    idx_list correct_trees_ijl(&arr_ijl[0],&arr_ijl[0]+1);
    size_t affected_loop_indices_arr[3] = {0,1,3};
    idx_list affected_loop_indices(&affected_loop_indices_arr[0],&affected_loop_indices_arr[0]+3);
    for(size_t i = 0; i < affected_loop_indices.size(); ++i)
    {
        idx_list trees = sf.get_trees_for_loop(affected_loop_indices[i]);
        if(trees != correct_trees_ijl)
        {
            __asm__ __volatile__("int $3"); 
            fail_test(test_name,__FILE__,__LINE__,
                    "sparsity_fuser::get_trees_for_loop(...) returned incorrect value after fusion");
        }
    } 

    /*idx_list trees_j = sf.get_trees_for_loop(1);*/
    /*size_t arr_j[1] = {0}; */
    /*idx_list correct_trees_j(&arr_j[0],&arr_j[0]+1);*/
    /*if(trees_j != correct_trees_j)*/
    /*{*/
        /*fail_test(test_name,__FILE__,__LINE__,*/
                /*"sparsity_fuser::get_trees_for_loop(...) returned incorrect value for j loop");*/
    /*}*/

    /*if(trees_l != correct_trees_l)*/
    /*{*/
        /*__asm__ __volatile__("int $3"); */
        /*fail_test(test_name,__FILE__,__LINE__,*/
                /*"sparsity_fuser::get_trees_for_loop(...) returned incorrect value for l loop");*/
    /*}*/
}

} // namespace libtensor
