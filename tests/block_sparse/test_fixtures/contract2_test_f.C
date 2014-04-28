#include "contract2_test_f.h"

namespace libtensor {

const size_t contract2_test_f::ij_sparsity[4][2] = {{0,0},
                                                    {0,1},
                                                    {1,1},
                                                    {1,2}};

const size_t contract2_test_f::kl_sparsity[4][2] = {{0,2},
                                                    {1,1},
                                                    {2,0},
                                                    {2,1}};

sparse_bispace<1> contract2_test_f::init_i()
{
    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);
    return spb_i;
}

sparse_bispace<1> contract2_test_f::init_j()
{
    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);
    return spb_j;
}

sparse_bispace<1> contract2_test_f::init_k()
{
    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);
    return spb_k;
}

sparse_bispace<1> contract2_test_f::init_l()
{
    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(2);
    split_points_l.push_back(5);
    spb_l.split(split_points_l);
    return spb_l;
}

const double contract2_test_f::s_A_arr[45] = { //i = 0 j = 0 k = 0
                                               1,2,
                                               //i = 0 j = 0 k = 1
                                               3,
                                               //i = 0 j = 0 k = 2
                                               4,5,

                                               //i = 0 j = 1 k = 0
                                               6,7,
                                               8,9,

                                               //i = 0 j = 1 k = 1
                                               10, 
                                               11,
                                               
                                               //i = 0 j = 1 k = 2
                                               12,13,
                                               14,15,

                                               //i = 1 j = 1 k = 0
                                               16,17,
                                               18,19,
                                               20,21,
                                               22,23,

                                               //i = 1 j = 1 k = 1
                                               24,
                                               25,
                                               26,
                                               27,

                                               //i = 1 j = 1 k = 2
                                               28,29,
                                               30,31,
                                               32,33,
                                               34,35,

                                               //i = 1 j = 2 k = 0
                                               36,37,
                                               38,39,

                                               //i = 1 j = 2 k = 1
                                               40,
                                               41,

                                               //i = 1 j = 2 k = 2
                                               42,43,
                                               44,45};

const double contract2_test_f::s_B_arr[60] = { //j = 0 k = 0 l = 2
                                               1,2,

                                               //j = 0 k = 1 l = 1
                                               3,4,5,

                                               //j = 0 k = 2 l = 0
                                               6,7,8,9,

                                               //j = 0 k = 2 l = 1
                                               10,11,12,13,14,15,

                                               //j = 1 k = 0 l = 2
                                               16,17,
                                               18,19,

                                               //j = 1 k = 1 l = 1
                                               20,21,22,
                                               23,24,25,

                                               //j = 1 k = 2 l = 0
                                               26,27,28,29,
                                               30,31,32,33,

                                               //j = 1 k = 2 l = 1
                                               34,35,36,37,38,39,
                                               40,41,42,43,44,45,

                                               //j = 2 k = 0 l = 2
                                               46,47,

                                               //j = 2 k = 1 l = 1
                                               48,49,50,

                                               //j = 2 k = 2 l = 0
                                               51,52,53,54,

                                               //j = 2 k = 2 l = 1
                                               55,56,57,58,59,60};
} // namespace libtensor
