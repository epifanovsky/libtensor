#include "permute_3d_sparse_120_test_f.h"

namespace libtensor {

const double permute_3d_sparse_120_test_f::s_input_arr[35] = { //k = 0, i = 0; j = 0
                         1,2,
                         3,4,
                         5,6,

                         //k = 0, i = 0, j = 1                        
                         7,8,9,
                         10,11,12,
                         13,14,15,

                         //k = 0, i = 1, j = 0
                         16,17,

                         //k = 1, i = 0, j = 0
                         21,22,
                         23,24,
                         25,26,
                         27,28,
                         29,30,
                         31,32,

                         //k = 1, i = 1, j = 1
                         55,56,57,
                         58,59,60};

const double permute_3d_sparse_120_test_f::s_output_arr[35] = { //i = 0 j = 0 k = 0
                              1,
                              2,
                              3,
                              4,
                              5,
                              6,


                              //i = 0 j = 0 k = 1
                              21,27,
                              22,28,
                              23,29,
                              24,30,
                              25,31,
                              26,32,

                              //i = 0 j = 1 k = 0
                              7,
                              8,
                              9,
                              10,
                              11,
                              12,
                              13,
                              14,
                              15,

                              //i = 1 j = 0 k = 0
                              16,
                              17,

                              // i = 1, j = 1 k = 1
                              55,58,
                              56,59,
                              57,60};

permutation<3> permute_3d_sparse_120_test_f::init_perm()
{
    return permutation<3>().permute(0,2).permute(0,1);
}

sparse_bispace<3> permute_3d_sparse_120_test_f::init_input_bispace()
{
    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1(1,1);
    spb_1.split(split_points_1);

    //Second bispace (mid index in input) and splitting
    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2(1,3);
    spb_2.split(split_points_2);

    //Third bispace (fast index in input) and splitting
    sparse_bispace<1> spb_3(5);
    std::vector<size_t> split_points_3(1,2);
    spb_3.split(split_points_3);

    //Sparsity data
    size_t sig_blocks_arr[5][3] = {{0,0,0},
                                   {0,0,1},
                                   {0,1,0},
                                   {1,0,0},
                                   {1,1,1}};
    std::vector< sequence<3,size_t> > sig_blocks(5);
    for(size_t entry_idx = 0; entry_idx < 5; ++entry_idx)
    {
        for(size_t subspace_idx = 0; subspace_idx < 3; ++subspace_idx)
        {
            sig_blocks[entry_idx][subspace_idx] = sig_blocks_arr[entry_idx][subspace_idx];
        }
    }
    /*sig_blocks[0][0] = 0; */
    /*sig_blocks[0][1] = 0;*/
    /*sig_blocks[0][2] = 0;*/
    /*sig_blocks[1][0] = 0; */
    /*sig_blocks[1][1] = 0;*/
    /*sig_blocks[1][2] = 1;*/
    /*sig_blocks[2][0] = 0; */
    /*sig_blocks[2][1] = 1;*/
    /*sig_blocks[2][2] = 0;*/
    /*sig_blocks[3][0] = 1; */
    /*sig_blocks[3][1] = 0;*/
    /*sig_blocks[3][2] = 0;*/
    /*sig_blocks[4][0] = 1; */
    /*sig_blocks[4][1] = 1;*/
    /*sig_blocks[4][2] = 1;*/

    sparse_bispace<3> three_d_input = spb_1 % spb_2 % spb_3 << sig_blocks;
    return three_d_input;
}

} // namespace libtensor
