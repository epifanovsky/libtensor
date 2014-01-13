/* * sparse_loop_list_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparse_loop_grouper.h>
#include "sparse_loop_grouper_test.h"

using namespace std;

namespace libtensor {

void sparse_loop_grouper_test::perform() throw(libtest::test_exception) {
    test_get_n_groups();
    test_get_bispaces_and_index_groups();
    /*test_get_offsets_and_sizes();*/
}

//Test fixtures
namespace {

//This test fixture checks that get_bispaces_and_index_groups and 
//get_offsets_and_sizes correctly include information for the dense tensors coupled to the primary tree
//D(ijk) = A(jil) Blm C(mk)
//The fully fused tree result is (index order ijklm): 
//  {1,2,4,4,5}; //offset 8,0,12
//  {1,2,4,5,5}; //offset 8,8,12
//  {4,5,2,3,3}; //offset 24,24,4
//  {5,2,1,5,2}; //offset 32,16,0
//  {5,2,4,5,5}; //offset 40,16,12
class dense_and_sparse_bispaces_test_f {
public:
    sparsity_fuser sf;

    //Reuse previous data structure as much as possible
    static sparsity_fuser init_sf()
    {
        vector< sparse_bispace_any_order > bispaces;
        vector< block_loop > loops;

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
        sparse_bispace<1> spb_m = spb_i;

        //Sparsity for D
        size_t key_0_arr_D[3] = {1,2,3}; //offset 0
        size_t key_1_arr_D[3] = {1,2,4}; //offset 8
        size_t key_2_arr_D[3] = {3,4,1}; //offset 16
        size_t key_3_arr_D[3] = {4,5,2}; //offset 24
        size_t key_4_arr_D[3] = {5,2,1}; //offset 32
        size_t key_5_arr_D[3] = {5,2,4}; //offset 40

        std::vector< sequence<3,size_t> > sig_blocks_D(6);
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[0][i] = key_0_arr_D[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[1][i] = key_1_arr_D[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[2][i] = key_2_arr_D[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[3][i] = key_3_arr_D[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[4][i] = key_4_arr_D[i];
        for(size_t i = 0; i < 3; ++i) sig_blocks_D[5][i] = key_5_arr_D[i];

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

        //Sparsity for C
        size_t key_0_arr_C[2] = {2,1}; //offset 0
        size_t key_1_arr_C[2] = {3,2}; //offset 4
        size_t key_2_arr_C[2] = {4,5}; //offset 8
        size_t key_3_arr_C[2] = {5,4}; //offset 12

        std::vector< sequence<2,size_t> > sig_blocks_C(4);
        for(size_t i = 0; i < 2; ++i) sig_blocks_C[0][i] = key_0_arr_C[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_C[1][i] = key_1_arr_C[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_C[2][i] = key_2_arr_C[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_C[3][i] = key_3_arr_C[i];

        sparse_bispace<3> spb_D = spb_i % spb_j % spb_k << sig_blocks_D;
        sparse_bispace<3> spb_A = spb_j % spb_i % spb_l << sig_blocks_A;
        sparse_bispace<2> spb_B = spb_l | spb_m;
        sparse_bispace<2> spb_C = spb_m % spb_k << sig_blocks_C;

        bispaces.push_back(spb_D);
        bispaces.push_back(spb_A);
        bispaces.push_back(spb_B);
        bispaces.push_back(spb_C);

        //Set up loop list
        loops.resize(5,block_loop(bispaces));
        //i loop
        loops[0].set_subspace_looped(0,0);
        loops[0].set_subspace_looped(1,1);

        //j loop
        loops[1].set_subspace_looped(0,1);
        loops[1].set_subspace_looped(1,0);

        //k loop
        loops[2].set_subspace_looped(0,2);
        loops[2].set_subspace_looped(3,1);

        //l loop
        loops[3].set_subspace_looped(1,2);
        loops[3].set_subspace_looped(2,0);

        //m loop
        loops[4].set_subspace_looped(2,1);
        loops[4].set_subspace_looped(3,0);

        sparsity_fuser sf(loops,bispaces);
        size_t fused_loops_arr_0[2] = {0,1};
        idx_list fused_loops(fused_loops_arr_0,fused_loops_arr_0+2);
        sf.fuse(0,1,fused_loops);

        size_t fused_loops_arr_1[1] = {2};
        idx_list fused_loops_1(fused_loops_arr_1,fused_loops_arr_1+1);
        sf.fuse(0,1,fused_loops_1);

        return sf; 
    }

    dense_and_sparse_bispaces_test_f() : sf(init_sf()) {}
};

} // namespace unnamed

void sparse_loop_grouper_test::test_get_n_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_n_groups()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(tf.sf);

    if(slg.get_n_groups() != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_n_groups(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_bispaces_and_index_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_bispaces_and_index_groups()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(tf.sf);

    vector<idx_pair_list> bispaces_and_index_groups = slg.get_bispaces_and_index_groups();

    vector<idx_pair_list> correct_bispaces_and_index_groups(1);
    correct_bispaces_and_index_groups[0].push_back(idx_pair(0,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(1,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(3,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(2,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(2,1));

    if(bispaces_and_index_groups != correct_bispaces_and_index_groups)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_offsets_and_sizes(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_offsets_and_sizes() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_offsets_and_sizes()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(tf.sf);

    vector< vector<off_dim_pair_list> > correct_oas(1);
    off_dim_pair correct_0_arr[5] = {off_dim_pair(8,8),off_dim_pair(0,8),off_dim_pair(12,4),off_dim_pair(8,2),off_dim_pair(10,2)};
    off_dim_pair correct_1_arr[5] = {off_dim_pair(8,8),off_dim_pair(8,8),off_dim_pair(12,4),off_dim_pair(10,2),off_dim_pair(10,2)};
    off_dim_pair correct_2_arr[5] = {off_dim_pair(24,8),off_dim_pair(24,8),off_dim_pair(4,4),off_dim_pair(6,2),off_dim_pair(6,2)};
    off_dim_pair correct_3_arr[5] = {off_dim_pair(32,8),off_dim_pair(16,8),off_dim_pair(0,4),off_dim_pair(10,2),off_dim_pair(4,2)};
    off_dim_pair correct_4_arr[5] = {off_dim_pair(40,8),off_dim_pair(16,8),off_dim_pair(12,4),off_dim_pair(10,2),off_dim_pair(10,2)};

    correct_oas[0].push_back(off_dim_pair_list(correct_0_arr,correct_0_arr+5));
    correct_oas[0].push_back(off_dim_pair_list(correct_1_arr,correct_1_arr+5));
    correct_oas[0].push_back(off_dim_pair_list(correct_2_arr,correct_2_arr+5));
    correct_oas[0].push_back(off_dim_pair_list(correct_3_arr,correct_3_arr+5));
    correct_oas[0].push_back(off_dim_pair_list(correct_4_arr,correct_4_arr+5));

    if(slg.get_offsets_and_sizes()  != correct_oas)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_offsets_and_sizes(...) returned incorrect value");
    }
}

} // namespace libtensor
