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
    test_get_offsets_and_sizes();
    test_fuse();
}

//Test fixtures
namespace {

//C_(ijk) = A_(jil) B_(lk)
class contract_test_f {
public:
    vector< sparse_bispace_any_order > bispaces; 
    vector< block_loop > loops;

    contract_test_f() 
    {
        //Set up bispaces - need 6 blocks of size 2 each
        sparse_bispace<1> spb_i(12);
        vector<size_t> split_points_i;
        for(size_t i = 2; i < 12; i += 2)
        {
            split_points_i.push_back(i);
        }
        spb_i.split(split_points_i);

        sparse_bispace<1> spb_j = spb_i;
        sparse_bispace<1> spb_k = spb_i;
        sparse_bispace<1> spb_l = spb_i;

        //Sparsity for C
        size_t key_0_arr_C[3] = {1,2,3}; //offset 0
        size_t key_1_arr_C[3] = {1,2,4}; //offset 8
        size_t key_2_arr_C[3] = {3,4,1}; //offset 16
        size_t key_3_arr_C[3] = {4,5,2}; //offset 24
        size_t key_4_arr_C[3] = {5,2,1}; //offset 32
        size_t key_5_arr_C[3] = {5,2,4}; //offset 40

        std::vector< sequence<3,size_t> > sig_blocks_C(6);
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[0][i] = key_0_arr_C[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[1][i] = key_1_arr_C[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[2][i] = key_2_arr_C[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[3][i] = key_3_arr_C[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[4][i] = key_4_arr_C[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_C[5][i] = key_5_arr_C[i];

        //Sparsity for A
        size_t key_0_arr_A[3] = {2,1,4}; //offset 0
        size_t key_1_arr_A[3] = {2,1,5}; //offset 8
        size_t key_2_arr_A[3] = {2,5,5}; //offset 16
        size_t key_3_arr_A[3] = {5,4,3}; //offset 24

        std::vector< sequence<3,size_t> > sig_blocks_A(4);
        for(size_t i = 0; i < 3; ++i) sig_blocks_A[0][i] = key_0_arr_A[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_A[1][i] = key_1_arr_A[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_A[2][i] = key_2_arr_A[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_A[3][i] = key_3_arr_A[i];

        //Sparsity for B
        size_t key_0_arr_B[2] = {2,1}; //offset 0
        size_t key_1_arr_B[2] = {3,2}; //offset 4
        size_t key_2_arr_B[2] = {4,5}; //offset 8
        size_t key_3_arr_B[2] = {5,4}; //offset 12

        std::vector< sequence<2,size_t> > sig_blocks_B(4);
        for(size_t i = 0; i < 2; ++i) sig_blocks_B[0][i] = key_0_arr_B[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_B[1][i] = key_1_arr_B[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_B[2][i] = key_2_arr_B[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_B[3][i] = key_3_arr_B[i];

        sparse_bispace<3> spb_C = spb_i % spb_j % spb_k << sig_blocks_C;
        sparse_bispace<3> spb_A = spb_j % spb_i % spb_l << sig_blocks_A;
        sparse_bispace<2> spb_B = spb_l % spb_k << sig_blocks_B;

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
    size_t arr_C[3] = {0,1,2}; 
    idx_list correct_li_C(arr_C,arr_C+3);
    if(loop_indices_C != correct_li_C)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_and_subspaces_for_tree(...) returned incorrect value for C sparse tree");
    }

    idx_list loop_indices_A = sf.get_loops_for_tree(1);
    size_t arr_A[3] = {0,1,3}; 
    idx_list correct_li_A(arr_A,arr_A+3);
    if(loop_indices_A != correct_li_A)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_and_subspaces_for_tree(...) returned incorrect value for A sparse tree");
    }

    idx_list loop_indices_B = sf.get_loops_for_tree(2);
    size_t arr_B[2] = {2,3}; 
    idx_list correct_li_B(arr_B,arr_B+2);
    if(loop_indices_B != correct_li_B)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_and_subspaces_for_tree(...) returned incorrect value for B sparse tree");
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
    size_t arr_k[2] = {0,2}; 
    idx_list correct_trees_k(arr_k,arr_k+2);
    if(trees_k != correct_trees_k)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for k loop");
    }

    idx_list trees_l = sf.get_trees_for_loop(3);
    size_t arr_l[2] = {1,2};
    idx_list correct_trees_l(arr_l,arr_l+2);
    if(trees_l != correct_trees_l)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for l loop");
    }
}

void sparsity_fuser_test::test_get_offsets_and_sizes() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_get_offset_and_sizes()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    vector<off_dim_pair_list> offsets_and_sizes = sf.get_offsets_and_sizes(0);
    vector<off_dim_pair_list> correct_oas;
    off_dim_pair correct_0_arr[1] = {off_dim_pair(0,8)  }; correct_oas.push_back(off_dim_pair_list(correct_0_arr,correct_0_arr+1));
    off_dim_pair correct_1_arr[1] = {off_dim_pair(8,8)  }; correct_oas.push_back(off_dim_pair_list(correct_1_arr,correct_1_arr+1));
    off_dim_pair correct_2_arr[1] = {off_dim_pair(16,8) }; correct_oas.push_back(off_dim_pair_list(correct_2_arr,correct_2_arr+1));
    off_dim_pair correct_3_arr[1] = {off_dim_pair(24,8) }; correct_oas.push_back(off_dim_pair_list(correct_3_arr,correct_3_arr+1));
    off_dim_pair correct_4_arr[1] = {off_dim_pair(32,8) }; correct_oas.push_back(off_dim_pair_list(correct_4_arr,correct_4_arr+1));
    off_dim_pair correct_5_arr[1] = {off_dim_pair(40,8) }; correct_oas.push_back(off_dim_pair_list(correct_5_arr,correct_5_arr+1));

    if(offsets_and_sizes != correct_oas)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_offsets_and_sizes(...) returned incorrect value C tree");
    }
}

void sparsity_fuser_test::test_fuse() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_fuser_test::test_fuse()";

    contract_test_f tf = contract_test_f();

    sparsity_fuser sf(tf.loops,tf.bispaces);

    /*** FUSE FIRST TWO TREES ***/
    size_t fused_loops_arr_0[2] = {0,1};
    idx_list fused_loops(fused_loops_arr_0,fused_loops_arr_0+2);
    sf.fuse(0,1,fused_loops);

    //CHECK LOOP->TREE MAPPING
    //The rhs tree should have disappeared from association with the relevant loops
    //The lhs tree should have been added to all loops with which the rhs tree was previously associated
    idx_list trees_i = sf.get_trees_for_loop(0);
    size_t arr_i[1] = {0}; 
    idx_list correct_trees_i(arr_i,arr_i+1);
    if(trees_i != correct_trees_i)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for i loop after fusion 0");
    }

    //i and j both refer to same single tree now 
    idx_list trees_j = sf.get_trees_for_loop(1);
    if(trees_j != correct_trees_i)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for j loop after fusion 0");
    }

    //Now that we only have two trees the B tree index should have declined by 1
    idx_list trees_k = sf.get_trees_for_loop(2);
    size_t arr_k[2] = {0,1};
    idx_list correct_trees_k(arr_k,arr_k+2);
    if(trees_k != correct_trees_k)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for k loop after fusion 0");
    }

    //l loop now also associated with the zero tree, making it the same as the k loop
    idx_list trees_l = sf.get_trees_for_loop(3);
    if(trees_l != correct_trees_k)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for l loop after fusion 0");
    }

    //CHECK LOOP->TREE MAPPING
    idx_list loops_0 = sf.get_loops_for_tree(0);
    size_t arr_0[4] = {0,1,2,3};
    idx_list correct_loops_0(arr_0,arr_0+4);
    if(loops_0 != correct_loops_0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_for_tree(...) returned incorrect value for tree 0 after fusion 0");
    }

    idx_list loops_1 = sf.get_loops_for_tree(1);
    size_t arr_1[2] = {2,3};
    idx_list correct_loops_1(arr_1,arr_1+2);
    if(loops_1 != correct_loops_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_for_tree(...) returned incorrect value for tree 1 after fusion 0");
    }

    //Check the tree entries
    vector<off_dim_pair_list> offsets_and_sizes_0 = sf.get_offsets_and_sizes(0);
    vector<off_dim_pair_list> correct_oas_0;
    off_dim_pair correct_0_arr_0[2] = {off_dim_pair(0,8),off_dim_pair(0,8) };   correct_oas_0.push_back(off_dim_pair_list(correct_0_arr_0,correct_0_arr_0+2));
    off_dim_pair correct_1_arr_0[2] = {off_dim_pair(0,8),off_dim_pair(8,8) };   correct_oas_0.push_back(off_dim_pair_list(correct_1_arr_0,correct_1_arr_0+2));
    off_dim_pair correct_2_arr_0[2] = {off_dim_pair(8,8),off_dim_pair(0,8) };   correct_oas_0.push_back(off_dim_pair_list(correct_2_arr_0,correct_2_arr_0+2));
    off_dim_pair correct_3_arr_0[2] = {off_dim_pair(8,8),off_dim_pair(8,8) };   correct_oas_0.push_back(off_dim_pair_list(correct_3_arr_0,correct_3_arr_0+2));
    off_dim_pair correct_4_arr_0[2] = {off_dim_pair(24,8),off_dim_pair(24,8) }; correct_oas_0.push_back(off_dim_pair_list(correct_4_arr_0,correct_4_arr_0+2));
    off_dim_pair correct_5_arr_0[2] = {off_dim_pair(32,8),off_dim_pair(16,8) }; correct_oas_0.push_back(off_dim_pair_list(correct_5_arr_0,correct_5_arr_0+2));
    off_dim_pair correct_6_arr_0[2] = {off_dim_pair(40,8),off_dim_pair(16,8) }; correct_oas_0.push_back(off_dim_pair_list(correct_6_arr_0,correct_6_arr_0+2));

    if(offsets_and_sizes_0 != correct_oas_0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_offsets_and_sizes(...) returned incorrect value for tree 0 after fusion 0");
    }

    /*** FUSE REMAINING TREE ***/
    size_t fused_loops_arr_1[2] = {2,3};
    idx_list fused_loops_1(fused_loops_arr_1,fused_loops_arr_1+2);
    sf.fuse(0,1,fused_loops_1);

    //CHECK LOOP->TREE MAPPING
    //All loops should now point to the single remaining tree 
    for(size_t loop_idx = 0; loop_idx < 4; ++loop_idx) 
    {
        idx_list trees = sf.get_trees_for_loop(loop_idx);
        if((trees.size() != 1) || (trees[0] != 0))
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparsity_fuser::get_trees_for_loop(...) returned incorrect value for a loop after fusion 1");
        }

    }

    //CHECK LOOP->TREE MAPPING
    //Should get same result as above
    loops_0 = sf.get_loops_for_tree(0);
    if(loops_0 != correct_loops_0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_loops_for_tree(...) returned incorrect value for tree 0 after fusion 1");
    }

    //Check the tree entries
    vector<off_dim_pair_list> offsets_and_sizes_1 = sf.get_offsets_and_sizes(0);
    vector<off_dim_pair_list> correct_oas_1;
    off_dim_pair correct_0_arr_1[3] = {off_dim_pair(8,8),off_dim_pair(8,8),off_dim_pair(12,4)};    correct_oas_1.push_back(off_dim_pair_list(correct_0_arr_1,correct_0_arr_1+3));
    off_dim_pair correct_1_arr_1[3] = {off_dim_pair(24,8),off_dim_pair(24,8),off_dim_pair(4,4)};   correct_oas_1.push_back(off_dim_pair_list(correct_1_arr_1,correct_1_arr_1+3));
    off_dim_pair correct_2_arr_1[3] = {off_dim_pair(40,8),off_dim_pair(16,8),off_dim_pair(12,4)};  correct_oas_1.push_back(off_dim_pair_list(correct_2_arr_1,correct_2_arr_1+3));

    if(offsets_and_sizes_1 != correct_oas_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_fuser::get_offsets_and_sizes(...) returned incorrect value for tree 0 after fusion 1");
    }
}

} // namespace libtensor
