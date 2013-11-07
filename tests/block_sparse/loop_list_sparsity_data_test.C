#include <libtensor/block_sparse/loop_list_sparsity_data.h>
#include <libtensor/block_sparse/sparse_bispace.h>
#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/runtime_permutation.h>
#include <libtensor/core/permutation.h>
#include "loop_list_sparsity_data_test.h"

namespace libtensor {

void loop_list_sparsity_data_test::perform() throw(libtest::test_exception) { 
    test_get_sig_block_list_in_order();
    /*test_get_sig_block_list_fuse_output_input();*/
}

//Simplest test: will not require weird weird offsets
void loop_list_sparsity_data_test::test_get_sig_block_list_in_order() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_sig_block_list_in_order()";

    //Create bispaces corresponding to 3d spare permutation
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(5);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);

    //Sparsity data
    std::vector< sequence<3,size_t> > sig_blocks(5);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 0;
    sig_blocks[0][2] = 0;
    sig_blocks[1][0] = 0; 
    sig_blocks[1][1] = 0;
    sig_blocks[1][2] = 1;
    sig_blocks[2][0] = 0; 
    sig_blocks[2][1] = 1;
    sig_blocks[2][2] = 0;
    sig_blocks[3][0] = 1; 
    sig_blocks[3][1] = 0;
    sig_blocks[3][2] = 0;
    sig_blocks[4][0] = 1; 
    sig_blocks[4][1] = 1;
    sig_blocks[4][2] = 1;
    sparse_bispace<3> three_d_input = spb_1 % spb_2 % spb_3 << sig_blocks;

    permutation<3> perm;
    perm.permute(0,2).permute(0,1);
    sparse_bispace<3> three_d_output = three_d_input.permute(perm);

    //Create loops corresponding to this permutation
    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<1,size_t> input_bispace_indices_1(1);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);


    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<1,size_t> input_bispace_indices_2(2);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    sequence<1,size_t> output_bispace_indices_3(2);
    sequence<1,size_t> input_bispace_indices_3(0);
    sequence<1,bool> output_ignore_3(false);
    sequence<1,bool> input_ignore_3(false);

    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
						input_bispace_indices_1,
						output_ignore_1,
						input_ignore_1));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
                        input_bispace_indices_2,
                        output_ignore_2,
                        input_ignore_2));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_3,
                        input_bispace_indices_3,
                        output_ignore_3,
                        input_ignore_3));

    sequence<1,sparse_bispace_any_order> output_bispaces(three_d_output);
    sequence<1,sparse_bispace_any_order> input_bispaces(three_d_input);

    std::vector<size_t> cur_block_idxs; 
    cur_block_idxs.push_back(0);
    cur_block_idxs.push_back(1);
    loop_list_sparsity_data llsd(loop_list,output_bispaces,input_bispaces);
    const block_list& my_block_list = llsd.get_sig_block_list(cur_block_idxs,2);

    //Correct answer: should be {0}
    const sparse_block_tree_any_order& output_tree =  three_d_output.get_sparse_group_tree(0);
    const block_list& correct_block_list = output_tree.get_sub_key_block_list(cur_block_idxs);

    if(my_block_list.size() != correct_block_list.size())
    {
            fail_test(test_name,__FILE__,__LINE__,
                    "loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output size");

    }
    for(size_t j = 0; j < correct_block_list.size(); ++j)
    {
        if(my_block_list[j] != correct_block_list[j])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output");
        }
    }
}

void loop_list_sparsity_data_test::test_get_sig_block_list_fuse_output_input() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_sig_block_list_fuse_output_input()";

    //bispace 1 - need 6 blocks
    sparse_bispace<1> spb_1(15);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    split_points_1.push_back(4);
    split_points_1.push_back(7);
    split_points_1.push_back(9);
    split_points_1.push_back(13);
    spb_1.split(split_points_1);

    //bispace 2 - need 6 blocks
    sparse_bispace<1> spb_2(15);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    split_points_2.push_back(5);
    split_points_2.push_back(8);
    split_points_2.push_back(12);
    split_points_2.push_back(14);
    spb_2.split(split_points_2);

    //Sparsity data 1
    size_t seq0_arr[2] = {1,2};
    size_t seq1_arr[2] = {1,5};
    size_t seq2_arr[2] = {2,3};
    size_t seq3_arr[2] = {4,1};
    size_t seq4_arr[2] = {4,4};
    size_t seq5_arr[2] = {5,1};
    size_t seq6_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > sig_blocks_1(7); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[0][i] = seq0_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[1][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[2][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[3][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[4][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[5][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[6][i] = seq6_arr[i];

    sparse_bispace<2> spb_C = spb_1 % spb_2 << sig_blocks_1;

    //Need 10 blocks
    sparse_bispace<1> spb_3(27);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(5);
    split_points_3.push_back(9);
    split_points_3.push_back(10);
    split_points_3.push_back(11);
    split_points_3.push_back(14);
    split_points_3.push_back(18);
    split_points_3.push_back(20);
    split_points_3.push_back(24);
    spb_3.split(split_points_3);

    //Sparsity data 2
    size_t seq0_arr_2[2] = {1,2};
    size_t seq1_arr_2[2] = {1,6};
    size_t seq2_arr_2[2] = {2,5};
    size_t seq3_arr_2[2] = {2,9};
    size_t seq4_arr_2[2] = {3,1};
    size_t seq5_arr_2[2] = {4,3};
    size_t seq6_arr_2[2] = {4,6};
    size_t seq7_arr_2[2] = {5,1};
    size_t seq8_arr_2[2] = {5,5};

    std::vector< sequence<2,size_t> > sig_blocks_2(9); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[0][i] = seq0_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[1][i] = seq1_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[2][i] = seq2_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[3][i] = seq3_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[4][i] = seq4_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[5][i] = seq5_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[6][i] = seq6_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[7][i] = seq7_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[8][i] = seq8_arr_2[i];

    sparse_bispace<2> spb_A = spb_1 % spb_3 << sig_blocks_2;

    //Sparsity data 3
    size_t seq00_arr_3[2] = {1,2};
    size_t seq01_arr_3[2] = {1,5};
    size_t seq02_arr_3[2] = {2,1};
    size_t seq03_arr_3[2] = {3,4};
    size_t seq04_arr_3[2] = {5,3};
    size_t seq05_arr_3[2] = {5,4};
    size_t seq06_arr_3[2] = {5,5};
    size_t seq07_arr_3[2] = {6,1};
    size_t seq08_arr_3[2] = {6,4};
    size_t seq09_arr_3[2] = {9,3};
    size_t seq10_arr_3[2] = {9,4};

    std::vector< sequence<2,size_t> > sig_blocks_3(11); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[0][i] = seq00_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[1][i] = seq01_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[2][i] = seq02_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[3][i] = seq03_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[4][i] = seq04_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[5][i] = seq05_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[6][i] = seq06_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[7][i] = seq07_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[8][i] = seq08_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[9][i] = seq09_arr_3[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_3[10][i] = seq10_arr_3[i];

    sparse_bispace<2> spb_B = spb_3 % spb_2 << sig_blocks_3;

    //Create loops corresponding to this matrix multiply
    //i loop 
    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<2,size_t> input_bispace_indices_1(0); //ignored in 
    sequence<1,bool> output_ignore_1(false);
    sequence<2,bool> input_ignore_1(false);
    input_ignore_1[1] = true;

    //j loop
    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<2,size_t> input_bispace_indices_2(1);
    sequence<1,bool> output_ignore_2(false);
    sequence<2,bool> input_ignore_2(false);
    input_ignore_2[0] = true;

    //k loop
    sequence<1,size_t> output_bispace_indices_3;
    sequence<2,size_t> input_bispace_indices_3(1);
    input_bispace_indices_3[1] = 0;
    sequence<1,bool> output_ignore_3(true);
    sequence<2,bool> input_ignore_3(false);

    std::vector< block_loop<1,2> > loop_list;
	loop_list.push_back(block_loop<1,2>(output_bispace_indices_1,
						input_bispace_indices_1,
						output_ignore_1,
						input_ignore_1));

    loop_list.push_back(block_loop<1,2>(output_bispace_indices_2,
                        input_bispace_indices_2,
                        output_ignore_2,
                        input_ignore_2));

    loop_list.push_back(block_loop<1,2>(output_bispace_indices_3,
                        input_bispace_indices_3,
                        output_ignore_3,
                        input_ignore_3));

    sequence<1,sparse_bispace_any_order> output_bispaces(spb_C);
    sequence<2,sparse_bispace_any_order> input_bispaces;
    input_bispaces[0] = spb_A;
    input_bispaces[1] = spb_B;

    loop_list_sparsity_data llsd(loop_list,output_bispaces,input_bispaces);

    //i = 4, j= 1
    std::vector<size_t> cur_block_idxs;
    cur_block_idxs.push_back(4);
    cur_block_idxs.push_back(1);
    const block_list& my_block_list = llsd.get_sig_block_list(cur_block_idxs,2);

    //Correct answer
    block_list correct_block_list(1,5); 
    if(my_block_list.size() != correct_block_list.size())
    {
        std::cout << "\nHERE!!!!!!\n";
        for(size_t j = 0; j < my_block_list.size(); ++j)
        {
            std::cout << my_block_list[j] << "\n";
        }
        fail_test(test_name,__FILE__,__LINE__,
                "loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output size");

    }
    for(size_t j = 0; j < correct_block_list.size(); ++j)
    {
        if(my_block_list[j] != correct_block_list[j])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output");
        }
    }
}

} // namespace libtensor
