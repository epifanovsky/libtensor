#include "contract2_dense_dense_test_f.h"

namespace libtensor {

const double contract2_dense_dense_test_f::s_A_arr[60] = {//i = 0 j = 0 k = 0 (1,2,2)
                                                          1,2,
                                                          3,4,

                                                          //i = 0 j = 0 k = 1 (1,2,3)
                                                          5,6,7,
                                                          8,9,10,

                                                          //i = 0 j = 1 k = 0 (1,2,2)
                                                          11,12,
                                                          13,14,

                                                          //i = 0 j = 1 k = 1 (1,2,3)
                                                          15,16,17,
                                                          18,19,20,

                                                          //i = 1 j = 0 k = 0 (2,2,2)
                                                          21,22,
                                                          23,24,
                                                          25,26,
                                                          27,28,

                                                          //i = 1 j = 0 k = 1 (2,2,3)
                                                          29,30,31,
                                                          32,33,34,
                                                          35,36,37,
                                                          38,39,40,

                                                          //i = 1 j = 1 k = 0 (2,2,2)
                                                          41,42,
                                                          43,44,
                                                          45,46,
                                                          47,48,


                                                          //i = 1 j = 1 k = 1 (2,2,3)
                                                          49,50,51,
                                                          52,53,54,
                                                          55,56,57,
                                                          58,59,60};




const double contract2_dense_dense_test_f::s_B_arr[30] = {//k = 0  l = 0
                                                          1,2,3,
                                                          4,5,6,

                                                          //k = 0 l = 1
                                                          7,8,9,
                                                          10,11,12,

                                                          //k = 1 l = 0
                                                          13,14,15,
                                                          16,17,18,
                                                          19,20,21,

                                                          //k = 1 l = 1
                                                          22,23,24,
                                                          25,26,27,
                                                          28,29,30};

const double contract2_dense_dense_test_f::s_C_correct_arr[72] = {//i = 0 j = 0 l = 0 
                                                                  303,324,345, 
                                                                  457,491,525,

                                                                  //i = 0 j = 0 l = 1
                                                                  483,504,525, 
                                                                  742,776,810,

                                                                  //i = 0 j = 1 l = 0 
                                                                  833,904,975,
                                                                  987,1071,1155,

                                                                  //i = 0 j = 1 l = 1
                                                                  1403,1474,1545,
                                                                  1662,1746,1830,

                                                                  //i = 1 j = 0 l = 0
                                                                  1555,1688,1821,
                                                                  1709,1855,2001,
                                                                  1863,2022,2181,
                                                                  2017,2189,2361,

                                                                  //i = 1 j = 0 l = 1
                                                                  2623,2756,2889,
                                                                  2882,3028,3174,
                                                                  3141,3300,3459,
                                                                  3400,3572,3744,

                                                                  //i = 1 j = 1 l = 0
                                                                  2615,2848,3081, 
                                                                  2769,3015,3261,
                                                                  2923,3182,3441,
                                                                  3077,3349,3621,

                                                                  //i = 1 j = 1 l = 1
                                                                  4463,4696,4929,
                                                                  4722,4968,5214, 
                                                                  4981,5240,5499,
                                                                  5240,5512,5784};

sparse_bispace<1> contract2_dense_dense_test_f::init_i()
{
    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);
    return spb_i;
}

sparse_bispace<1> contract2_dense_dense_test_f::init_j()
{
    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
    spb_j.split(split_points_j);
    return spb_j;
}

sparse_bispace<1> contract2_dense_dense_test_f::init_k()
{
    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    spb_k.split(split_points_k);
    return spb_k;
}

sparse_bispace<1> contract2_dense_dense_test_f::init_l()
{
    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(3);
    spb_l.split(split_points_l);
    return spb_l;
}

} // namespace libtensor
