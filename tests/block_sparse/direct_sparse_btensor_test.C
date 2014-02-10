#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/contract.h>
#include "direct_sparse_btensor_test.h"

namespace libtensor {

void direct_sparse_btensor_test::perform() throw(libtest::test_exception) {
    test_get_batch_contract2();
    test_contract2_direct_rhs();
    test_contract2_subtract2_nested();
}

void direct_sparse_btensor_test::test_get_batch_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_get_batch_contract2()";

    //Block major
    double A_arr[45] = { //i = 0 j = 0 k = 0
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
    double B_arr[60] = {  //j = 0 k = 0 l = 2
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


    double C_batch_0_correct_arr[6] = {  //i = 0 l = 0
                                         1640,1703,
                                         
                                         //i = 0 l = 1
                                         2661,2748,2835,

                                         //i = 0 l = 2
                                         535,
                                         };

    double C_batch_1_correct_arr[12] = { //i = 1 l = 0
                                         7853,8056,
                                         8525,8748,
                                         
                                         
                                         //i = 1 l = 1
                                         12337,12629,12921,
                                         13313,13630,13947,
                                         
                                         //i = 1 l = 2
                                         4625,
                                         5091
                                         };


    //Alloc enough memory to hold the largest batch
    double C_batch_arr[12] = {0};

    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);

    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(2);
    split_points_l.push_back(5);
    spb_l.split(split_points_l);


    //(ij) sparsity
    size_t seq_00_arr_1[2] = {0,0};
    size_t seq_01_arr_1[2] = {0,1};
    size_t seq_02_arr_1[2] = {1,1};
    size_t seq_03_arr_1[2] = {1,2};

    std::vector< sequence<2,size_t> > ij_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[0][i] = seq_00_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[1][i] = seq_01_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[2][i] = seq_02_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[3][i] = seq_03_arr_1[i];

    sparse_bispace<3> spb_A = spb_i % spb_j << ij_sig_blocks | spb_k;


    //(kl) sparsity
    size_t seq_00_arr_2[2] = {0,2};
    size_t seq_01_arr_2[2] = {1,1};
    size_t seq_02_arr_2[2] = {2,0};
    size_t seq_03_arr_2[2] = {2,1};

    std::vector< sequence<2,size_t> > kl_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[0][i] = seq_00_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[1][i] = seq_01_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[2][i] = seq_02_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[3][i] = seq_03_arr_2[i];

    sparse_bispace<3> spb_B = spb_j | spb_k % spb_l << kl_sig_blocks;

    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_bispace<2> spb_C = spb_i | spb_l;

    direct_sparse_btensor<2> C(spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    std::map<idx_pair,idx_pair> batches;
    batches[idx_pair(0,0)] = idx_pair(0,1);
    C.get_batch(C_batch_arr,batches);

    for(size_t i = 0; i < sizeof(C_batch_0_correct_arr)/sizeof(C_batch_0_correct_arr[0]); ++i)
    {
        if(C_batch_arr[i] != C_batch_0_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "contract(...) did not produce correct result for batch 0");
        }
    }
    
    batches[idx_pair(0,0)] = idx_pair(1,2);
    C.get_batch(C_batch_arr,batches);
    for(size_t i = 0; i < sizeof(C_batch_1_correct_arr)/sizeof(C_batch_1_correct_arr[0]); ++i)
    {
        if(C_batch_arr[i] != C_batch_1_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "contract(...) did not produce correct result for batch 1");
        }
    }
}

//TODO: group with other test in test fixture
void direct_sparse_btensor_test::test_contract2_direct_rhs() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_direct_rhs()";

    //Block major
    double A_arr[45] = { //i = 0 j = 0 k = 0
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
    double B_arr[60] = {  //j = 0 k = 0 l = 2
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



    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);

    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(2);
    split_points_l.push_back(5);
    spb_l.split(split_points_l);

    //(ij) sparsity
    size_t seq_00_arr_1[2] = {0,0};
    size_t seq_01_arr_1[2] = {0,1};
    size_t seq_02_arr_1[2] = {1,1};
    size_t seq_03_arr_1[2] = {1,2};

    std::vector< sequence<2,size_t> > ij_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[0][i] = seq_00_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[1][i] = seq_01_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[2][i] = seq_02_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[3][i] = seq_03_arr_1[i];

    sparse_bispace<3> spb_A = spb_i % spb_j << ij_sig_blocks | spb_k;


    //(kl) sparsity
    size_t seq_00_arr_2[2] = {0,2};
    size_t seq_01_arr_2[2] = {1,1};
    size_t seq_02_arr_2[2] = {2,0};
    size_t seq_03_arr_2[2] = {2,1};

    std::vector< sequence<2,size_t> > kl_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[0][i] = seq_00_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[1][i] = seq_01_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[2][i] = seq_02_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[3][i] = seq_03_arr_2[i];



    sparse_bispace<3> spb_B = spb_j | spb_k % spb_l << kl_sig_blocks;

    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_bispace<2> spb_C = spb_i | spb_l;

    direct_sparse_btensor<2> C(spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    /*** SECOND STEP - USE DIRECT TENSOR ***/

    //(ml) sparsity
    size_t seq_0_arr_ml[2] = {0,1};
    size_t seq_1_arr_ml[2] = {0,2};
    size_t seq_2_arr_ml[2] = {1,0};
    size_t seq_3_arr_ml[2] = {1,2};

    std::vector< sequence<2,size_t> > ml_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[0][i] = seq_0_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[1][i] = seq_1_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[2][i] = seq_2_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[3][i] = seq_3_arr_ml[i];

    //Bispace for m
    sparse_bispace<1> spb_m(6);
    std::vector<size_t> split_points_m;
    split_points_m.push_back(3);
    spb_m.split(split_points_m);


    sparse_bispace<2> spb_D = spb_m % spb_l << ml_sig_blocks;
    double D_arr[21] = {  //m = 0 l = 1
                          1,2,3,
                          4,5,6,
                          7,8,9,

                          //m = 0 l = 2
                          10,
                          11,
                          12,

                          //m = 1 l = 0
                          13,14,
                          15,16,
                          17,18,

                          //m = 1 l = 2
                          19,
                          20,
                          21
                          };

    sparse_btensor<2> D(spb_D,D_arr,true);

    sparse_bispace<2> spb_E = spb_m | spb_i;
    sparse_btensor<2> E(spb_E);
    letter m;

    //Make batch memory just big enough to fit i = 1 batch of C 
    //This will force partitioning into i = 0 and i = 1
    E(m|i) = contract(l,D(m|l),C(i|l),96);

    double E_correct_arr[18] = { //m = 0 i = 0
                                 22012,
                                 47279,
                                 72546,

                                 //m = 0 i = 1
                                 122608,133324,
                                 240894,261085,
                                 359180,388846,

                                 //m = 1 i = 0
                                 55327,
                                 62548,
                                 69769,

                                 //m = 1 i = 1
                                 302748,330026,
                                 339191,369663,
                                 375634,409300
                               };


    sparse_btensor<2> E_correct(spb_E,E_correct_arr,true);
    if(E != E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void direct_sparse_btensor_test::test_contract2_subtract2_nested() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_subtract2_nested()";

    //Block major
    double A_arr[45] = { //i = 0 j = 0 k = 0
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
    double B_arr[60] = {  //j = 0 k = 0 l = 2
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



    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);

    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(2);
    split_points_l.push_back(5);
    spb_l.split(split_points_l);

    //(ij) sparsity
    size_t seq_00_arr_1[2] = {0,0};
    size_t seq_01_arr_1[2] = {0,1};
    size_t seq_02_arr_1[2] = {1,1};
    size_t seq_03_arr_1[2] = {1,2};

    std::vector< sequence<2,size_t> > ij_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[0][i] = seq_00_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[1][i] = seq_01_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[2][i] = seq_02_arr_1[i];
    for(size_t i = 0; i < 2; ++i) ij_sig_blocks[3][i] = seq_03_arr_1[i];

    sparse_bispace<3> spb_A = spb_i % spb_j << ij_sig_blocks | spb_k;


    //(kl) sparsity
    size_t seq_00_arr_2[2] = {0,2};
    size_t seq_01_arr_2[2] = {1,1};
    size_t seq_02_arr_2[2] = {2,0};
    size_t seq_03_arr_2[2] = {2,1};

    std::vector< sequence<2,size_t> > kl_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[0][i] = seq_00_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[1][i] = seq_01_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[2][i] = seq_02_arr_2[i];
    for(size_t i = 0; i < 2; ++i) kl_sig_blocks[3][i] = seq_03_arr_2[i];



    sparse_bispace<3> spb_B = spb_j | spb_k % spb_l << kl_sig_blocks;

    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<3> B(spb_B,B_arr,true);
    sparse_bispace<2> spb_C = spb_i | spb_l;

    direct_sparse_btensor<2> C(spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    /*** SECOND STEP - SUBTRACTION OF DIRECT TENSOR ***/
    double F_arr[18] = { //i = 0 l = 0
                         1,2,
                         
                         //i = 0 l = 1
                         3,4,5,

                         //i = 0 l = 2
                         6,

                         //i = 1 l = 0
                         7,8,
                         9,10,
                         
                         //i = 1 l = 1
                         11,12,13,
                         14,15,16,
                         
                         //i = 1 l = 2
                         17,
                         18
                        };

    sparse_btensor<2> F(spb_C,F_arr,true);
    direct_sparse_btensor<2> G(spb_C);

    G(i|l) = C(i|l) - F(i|l);


    /*** SECOND STEP - USE DIRECT TENSOR ***/

    //(ml) sparsity
    size_t seq_0_arr_ml[2] = {0,1};
    size_t seq_1_arr_ml[2] = {0,2};
    size_t seq_2_arr_ml[2] = {1,0};
    size_t seq_3_arr_ml[2] = {1,2};

    std::vector< sequence<2,size_t> > ml_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[0][i] = seq_0_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[1][i] = seq_1_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[2][i] = seq_2_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[3][i] = seq_3_arr_ml[i];

    //Bispace for m
    sparse_bispace<1> spb_m(6);
    std::vector<size_t> split_points_m;
    split_points_m.push_back(3);
    spb_m.split(split_points_m);


    sparse_bispace<2> spb_D = spb_m % spb_l << ml_sig_blocks;
    double D_arr[21] = {  //m = 0 l = 1
                          1,2,3,
                          4,5,6,
                          7,8,9,

                          //m = 0 l = 2
                          10,
                          11,
                          12,

                          //m = 1 l = 0
                          13,14,
                          15,16,
                          17,18,

                          //m = 1 l = 2
                          19,
                          20,
                          21
                          };

    sparse_btensor<2> D(spb_D,D_arr,true);

    sparse_bispace<2> spb_E = spb_m | spb_i;
    sparse_btensor<2> E(spb_E);
    letter m;

    //Make batch memory just big enough to fit i = 1 batch of C 
    //This will force partitioning into i = 0 and i = 1
    E(m|i) = contract(l,D(m|l),G(i|l),96);

    double E_correct_arr[18] = { //m = 0 i = 0
                                 21926,47151,72376,

                                 //m = 0 i = 1
                                 122364,133052, 
                                 240525,260660,
                                 358686,388268,

                                 //m = 1 i = 0
                                 55172,
                                 62381,
                                 69590,

                                 //m = 1 i = 1
                                 302222,329427,
                                 338618,369008,
                                 375014,408589
                               };


    sparse_btensor<2> E_correct(spb_E,E_correct_arr,true);
    if(E != E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

} // namespace libtensor
