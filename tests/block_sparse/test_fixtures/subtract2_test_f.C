#include "subtract2_test_f.h"

namespace libtensor {

const size_t subtract2_test_f::ij_sparsity[4][2] = {{0,0},
                                                    {0,1},
                                                    {1,1},
                                                    {1,2}};


sparse_bispace<1> subtract2_test_f::init_i()
{
    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);
    return spb_i;
}

sparse_bispace<1> subtract2_test_f::init_j()
{
    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);
    return spb_j;
}

sparse_bispace<1> subtract2_test_f::init_k()
{
    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);
    return spb_k;
}


//Block major
const double subtract2_test_f::s_A_arr[45] = { //i = 0 j = 0 k = 0
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

//Block major
const double subtract2_test_f::s_B_arr[60] = { //i = 0 j = 0 k = 0
                                               2,4,
                                               //i = 0 j = 0 k = 1
                                               6,
                                               //i = 0 j = 0 k = 2
                                               8,10,

                                               //i = 0 j = 1 k = 0
                                               12,14,
                                               16,18,

                                               //i = 0 j = 1 k = 1
                                               20,
                                               22,
                    
                                               //i = 0 j = 1 k = 2
                                               24,26,
                                               28,30,

                                               //i = 0 j = 2 k = 0
                                               32,34,

                                               //i = 0 j = 2 k = 1
                                               36,

                                               //i = 0 j = 2 k = 2
                                               38,40,

                                               //i = 1 j = 0 k = 0
                                               42,44,
                                               46,48,

                                               //i = 1 j = 0 k = 1
                                               50,
                                               52,

                                               //i = 1 j = 0 k = 2
                                               54,56,
                                               58,60,

                                               //i = 1 j = 1 k = 0
                                               62,64,
                                               66,68,
                                               70,72,
                                               74,76,

                                               //i = 1 j = 1 k = 1
                                               78,
                                               80,
                                               82,
                                               84,

                                               //i = 1 j = 1 k = 2
                                               86,88,
                                               90,92,
                                               94,96,
                                               98,100,

                                               //i = 1 j = 2 k = 0
                                               102,104,
                                               106,108,

                                               //i = 1 j = 2 k = 1
                                               110,
                                               112,

                                               //i = 1 j = 2 k = 2
                                               114,116,
                                               118,120};

//Block major
const double subtract2_test_f::s_C_arr[60] = { //i = 0 j = 0 k = 0
                                               -1,-2,
                                               //i = 0 j = 0 k = 1
                                               -3,
                                               //i = 0 j = 0 k = 2
                                               -4,-5,

                                               //i = 0 j = 1 k = 0
                                               -6,-7,
                                               -8,-9,

                                               //i = 0 j = 1 k = 1
                                               -10,
                                               -11,
                            
                                               //i = 0 j = 1 k = 2
                                               -12,-13,
                                               -14,-15,

                                               //i = 0 j = 2 k = 0
                                               -32,-34,

                                               //i = 0 j = 2 k = 1
                                               -36,

                                               //i = 0 j = 2 k = 2
                                               -38,-40,

                                               //i = 1 j = 0 k = 0
                                               -42,-44,
                                               -46,-48,

                                               //i = 1 j = 0 k = 1
                                               -50,
                                               -52,

                                               //i = 1 j = 0 k = 2
                                               -54,-56,
                                               -58,-60,

                                               //i = 1 j = 1 k = 0
                                               -46,-47,
                                               -48,-49,
                                               -50,-51,
                                               -52,-53,

                                               //i = 1 j = 1 k = 1
                                               -54,
                                               -55,
                                               -56,
                                               -57,

                                               //i = 1 j = 1 k = 2
                                               -58,-59, 
                                               -60,-61,
                                               -62,-63,
                                               -64,-65,

                                               //i = 1 j = 2 k = 0
                                               -66,-67,
                                               -68,-69,

                                               //i = 1 j = 2 k = 1
                                               -70,
                                               -71,

                                               //i = 1 j = 2 k = 2
                                               -72,-73,
                                               -74,-75};
} // namespace libtensor
