#include <libtensor/block_sparse/loop_list_sparsity_data.h>
#include <libtensor/block_sparse/loop_list_sparsity_data_new.h>
#include <libtensor/block_sparse/sparse_bispace.h>
#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/runtime_permutation.h>
#include <libtensor/core/permutation.h>
#include "loop_list_sparsity_data_test.h"

namespace libtensor {

void loop_list_sparsity_data_test::perform() throw(libtest::test_exception) { 
    test_get_sig_block_list_no_sparsity();
    test_get_sig_block_list_sparsity_one_tensor();
    test_get_sig_block_list_sparsity_3_tensors();
}

//kij -> ijk, no sparsity
void loop_list_sparsity_data_test::test_get_sig_block_list_no_sparsity() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_sig_block_list_no_sparsity()";

    //Create bispaces corresponding to 3d dense permutation
    sparse_bispace<1> spb_1(8);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    split_points_1.push_back(4);
    split_points_1.push_back(6);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(15);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    split_points_2.push_back(5);
    split_points_2.push_back(8);
    split_points_2.push_back(12);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(12);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(3);
    split_points_3.push_back(7);
    split_points_3.push_back(10);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces(1,spb_1|spb_2|spb_3);
    bispaces.push_back(spb_3|spb_1|spb_2);

    block_loop_new bl_1(bispaces);
    bl_1.set_subspace_looped(0,0);
    bl_1.set_subspace_looped(1,1);
    block_loop_new bl_2(bispaces);
    bl_2.set_subspace_looped(0,1);
    bl_2.set_subspace_looped(1,2);
    block_loop_new bl_3(bispaces);
    bl_3.set_subspace_looped(0,2);
    bl_3.set_subspace_looped(1,0);

    sparse_loop_list sll(bispaces);
    sll.add_loop(bl_1);
    sll.add_loop(bl_2);
    sll.add_loop(bl_3);

    //Loop block indices chosen should be irrelevant, should return the full block range of spb_3
    loop_list_sparsity_data_new llsd(sll);
    block_list loop_block_inds;
    loop_block_inds.push_back(2);
    loop_block_inds.push_back(3);
    block_list bl = llsd.get_sig_block_list(loop_block_inds,2);

    //Correct answer
    block_list bl_correct = range(0,5);

    if(bl.size() != bl_correct.size())
    {
		std::cout << "\nmy size: " << bl.size() << "\n";
		std::cout << "\ncorrect size: " << bl_correct.size() << "\n";
		fail_test(test_name,__FILE__,__LINE__,
				"loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output size");
    }

    for(size_t i  = 0; i < bl.size(); ++i)
    {
    	if(bl[i] != bl_correct[i])
    	{
			fail_test(test_name,__FILE__,__LINE__,
					"loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output");
    	}
    }
}

void loop_list_sparsity_data_test::test_get_sig_block_list_sparsity_one_tensor() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_sig_block_list_sparsity_one_tensor()";

    //Create bispaces corresponding to 3d sparse permutation
    //Need 8 blocks
    sparse_bispace<1> spb_1(20);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    split_points_1.push_back(4);
    split_points_1.push_back(6);
    split_points_1.push_back(9);
    split_points_1.push_back(13);
    split_points_1.push_back(14);
    split_points_1.push_back(16);
    spb_1.split(split_points_1);

    //Need 8 blocks
    sparse_bispace<1> spb_2(24);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    split_points_2.push_back(5);
    split_points_2.push_back(8);
    split_points_2.push_back(12);
    split_points_2.push_back(16);
    split_points_2.push_back(17);
    split_points_2.push_back(19);
    split_points_2.push_back(22);
    spb_2.split(split_points_2);

    //Need 8 blocks
    sparse_bispace<1> spb_3(21);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(3);
    split_points_3.push_back(7);
    split_points_3.push_back(10);
    split_points_3.push_back(15);
    split_points_3.push_back(16);
    split_points_3.push_back(18);
    spb_3.split(split_points_3);


    //Sparsity data
    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,7};
    size_t seq05_arr[3] = {2,3,1};
    size_t seq06_arr[3] = {2,4,2};
    size_t seq07_arr[3] = {2,4,5};
    size_t seq08_arr[3] = {2,6,3};
    size_t seq09_arr[3] = {2,6,4};
    size_t seq10_arr[3] = {4,1,4};
    size_t seq11_arr[3] = {4,1,7};
    size_t seq12_arr[3] = {4,2,2};
    size_t seq13_arr[3] = {4,3,5};
    size_t seq14_arr[3] = {4,3,6};
    size_t seq15_arr[3] = {4,3,7};
    size_t seq16_arr[3] = {5,1,4};
    size_t seq17_arr[3] = {5,2,6};
    size_t seq18_arr[3] = {5,2,7};
    size_t seq19_arr[3] = {7,4,5};
    size_t seq20_arr[3] = {7,4,6};
    size_t seq21_arr[3] = {7,7,7};

    std::vector< sequence<3,size_t> > sig_blocks(21);
    for(size_t i = 0; i < 3; ++i) sig_blocks[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[20][i] = seq21_arr[i];

    std::vector<sparse_bispace_any_order> bispaces(1,spb_1 % spb_2 % spb_3 << sig_blocks);

    block_loop_new bl_1(bispaces);
    bl_1.set_subspace_looped(0,0);
    block_loop_new bl_2(bispaces);
    bl_2.set_subspace_looped(0,1);
    block_loop_new bl_3(bispaces);
    bl_3.set_subspace_looped(0,2);

    sparse_loop_list sll(bispaces);
    sll.add_loop(bl_1);
    sll.add_loop(bl_2);
    sll.add_loop(bl_3);

    loop_list_sparsity_data_new llsd(sll);

    std::vector<size_t> loop_block_indices(1,4);
    loop_block_indices.push_back(3);
    block_list bl = llsd.get_sig_block_list(loop_block_indices,2);

    //Correct answer: {5,6,7}
    block_list bl_correct(1,5);
    bl_correct.push_back(6);
    bl_correct.push_back(7);

    if(bl.size() != bl_correct.size())
    {
		fail_test(test_name,__FILE__,__LINE__,
				"loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output size");
    }

    for(size_t i  = 0; i < bl.size(); ++i)
    {
    	if(bl[i] != bl_correct[i])
    	{
			fail_test(test_name,__FILE__,__LINE__,
					"loop_list_sparsity_data::get_sig_block_list(...) produced incorrect output");
    	}
    }
}

void loop_list_sparsity_data_test::test_get_sig_block_list_sparsity_3_tensors() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_sig_block_list_sparsity_3_tensors()";

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

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1 % spb_2 << sig_blocks_1);

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

    bispaces.push_back(spb_1 % spb_3 << sig_blocks_2);

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

    bispaces.push_back(spb_3 % spb_2 << sig_blocks_3);

    //Create loops corresponding to this matrix multiply
    //i loop 
    block_loop_new bl_1(bispaces);
    bl_1.set_subspace_looped(0,0);
    bl_1.set_subspace_looped(1,0);
    //j loop
    block_loop_new bl_2(bispaces);
    bl_2.set_subspace_looped(0,1);
    bl_2.set_subspace_looped(2,1);
    //k loop
    block_loop_new bl_3(bispaces);
    bl_3.set_subspace_looped(1,1);
    bl_3.set_subspace_looped(2,0);

    sparse_loop_list sll(bispaces);
    sll.add_loop(bl_1);
    sll.add_loop(bl_2);
    sll.add_loop(bl_3);

    std::cout << "######################### HERE ####################\n";
    loop_list_sparsity_data_new llsd(sll);

    //i = 4, j= 1
    std::vector<size_t> loop_block_indices;
    loop_block_indices.push_back(4);
    loop_block_indices.push_back(1);
    const block_list& my_block_list = llsd.get_sig_block_list(loop_block_indices,2);

    //Correct answer
    block_list correct_block_list(1,6); 
    if(my_block_list.size() != correct_block_list.size())
    {
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
