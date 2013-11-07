
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/contract.h>
#include <libtensor/iface/letter.h>
#include "sparse_btensor_test.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {
   
void sparse_btensor_test::perform() throw(libtest::test_exception) {

    test_get_bispace();

    test_str_2d_block_major();
    test_str_2d_row_major();
    test_str_3d_row_major();

    test_equality_different_nnz();
    test_equality_true();
    test_equality_false();

    test_permute_2d_row_major();
    test_permute_3d_row_major_210();
    /*test_permute_3d_block_major_120_sparse();*/

    test_contract2_2d_2d();
    test_contract2_2d_2d_sparse_dense(); 
    test_contract2_3d_2d_sparse_dense();
    test_contract2_3d_2d_sparse_sparse();
    test_contract2_two_indices_3d_2d_dense_dense();

    test_contract2_3d_2d();
}

void sparse_btensor_test::test_get_bispace() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_get_bispace()";
    sparse_bispace<1> spb_1(5);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d = spb_1 | spb_2;
    sparse_btensor<2> sbt(two_d);
    if(sbt.get_bispace() != two_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::get_bispace(...) did not return two_d");
    }
}

void sparse_btensor_test::test_str_2d_block_major() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_str_2d_block_major()";
    double mem_block_major[16] = { 1,2,5,6,
                                   3,4,7,8,
                                   9,10,13,14,
                                  11,12,15,16};


    sparse_bispace<1> N(4);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;
    sparse_btensor<2> bt(N2,mem_block_major,true);

    std::string correct_str("---\n 1 2\n 5 6\n---\n 3 4\n 7 8\n---\n 9 10\n 13 14\n---\n 11 12\n 15 16\n");
    if(bt.str() != correct_str)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::str(...) did not return correct string");
    }
}

void sparse_btensor_test::test_str_2d_row_major() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_str_2d_row_major()";
    double mem_row_major[16] = { 1,2,3,4,
                                 5,6,7,8,
                                 9,10,11,12,
                                 13,14,15,16};


    sparse_bispace<1> N(4);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;
    sparse_btensor<2> bt(N2,mem_row_major);

    std::string correct_str("---\n 1 2\n 5 6\n---\n 3 4\n 7 8\n---\n 9 10\n 13 14\n---\n 11 12\n 15 16\n");
    if(bt.str() != correct_str)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::str(...) did not return correct string");
    }
}

void sparse_btensor_test::test_str_3d_row_major() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_str_3d_row_major()";
    
    //3x4x5
    //kij -> ijk permutation
    //indices are canonical (not block) indices
    double mem_row_major[60] = { //k = 0 
                                 1,2,3,4,5, //i = 0
                                 6,7,8,9,10, //i = 1
                                 11,12,13,14,15, //i = 2
                                 16,17,18,19,20, //i = 3

                                 //k = 1
                                 21,22,23,24,25, //i = 0
                                 26,27,28,29,30, //i = 1
                                 31,32,33,34,35, //i = 2 
                                 36,37,38,39,40, //i = 3

                                 //k = 2
                                 41,42,43,44,45,
                                 46,47,48,49,50,
                                 51,52,53,54,55,
                                 56,57,58,59,60};

    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    //Second bispace (mid index in input) and splitting
    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    spb_2.split(split_points_2);

    //Third bispace (fast index in input) and splitting
    sparse_bispace<1> spb_3(5);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);

    sparse_bispace<3> three_d_input = spb_1 | spb_2 | spb_3;

    sparse_btensor<3> bt(three_d_input,mem_row_major);

    std::string correct_str("---\n"
                            " 1 2\n"
                            " 6 7\n"
                            " 11 12\n"
                            "---\n"
                            " 3 4 5\n"
                            " 8 9 10\n"
                            " 13 14 15\n"
                            "---\n"
                            " 16 17\n"
                            "---\n"
                            " 18 19 20\n"
                            "---\n"
                            " 21 22\n"
                            " 26 27\n"
                            " 31 32\n\n"
                            " 41 42\n"
                            " 46 47\n"
                            " 51 52\n"
                            "---\n"
                            " 23 24 25\n"
                            " 28 29 30\n"
                            " 33 34 35\n\n"
                            " 43 44 45\n"
                            " 48 49 50\n"
                            " 53 54 55\n"
                            "---\n"
                            " 36 37\n\n"
                            " 56 57\n"
                            "---\n"
                            " 38 39 40\n\n"
                            " 58 59 60\n");

    if(bt.str() != correct_str)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::str(...) did not return correct string");
    }
}

void sparse_btensor_test::test_equality_different_nnz() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_equality_different_nnz()";

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

    sparse_bispace<2> two_d_1 = spb_1 | spb_2;
    sparse_bispace<2> two_d_2 = spb_1 | spb_3;

    sparse_btensor<2> bt_1(two_d_1);
    sparse_btensor<2> bt_2(two_d_2);


    bool threw_exception = false;
    try
    {
        bt_1 == bt_2;
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::operator==(...) did not throw exception");
    }
}

void sparse_btensor_test::test_equality_true() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_equality_true()";

    double mem_block_major[16] = { 1,2,
                                   5,6,

                                   3,4,
                                   7,8,

                                   9,10, 
                                   13,14,

                                   11,12,
                                   15,16};

    double mem_row_major[16] = { 1,2,3,4,
                                 5,6,7,8,
                                 9,10,11,12,
                                 13,14,15,16};


    sparse_bispace<1> N(4);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;

    sparse_btensor<2> block_major(N2,mem_block_major,true);
    sparse_btensor<2> row_major(N2,mem_row_major);

    if(!(row_major == block_major))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::operator==(...) did not return true");
    }
}

void sparse_btensor_test::test_equality_false() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_equality_false()";

    double mem_block_major[16] = { 1,2,
                                   5,6,

                                   3,4,
                                   7,8,

                                   9,10, 
                                   13,14,

                                   11,12,
                                   15,16};

    double mem_row_major[16] = { 1,2,3,4,
                                 5,6,7,8,
                                 9,10,11,12,
                                 13,14,15,17}; //Last entry changed from 16->17


    sparse_bispace<1> N(4);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;

    sparse_btensor<2> block_major(N2,mem_block_major,true);
    sparse_btensor<2> row_major(N2,mem_row_major);

    if(row_major == block_major)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::operator==(...) did not return false");
    }
}

void sparse_btensor_test::test_permute_2d_row_major() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_permute_2d_row_major()";

    double mem_row_major[20] = { 1,2,3,4,5,
                                 6,7,8,9,10,
                                 11,12,13,14,15,
                                 16,17,18,19,20}; 

    
    //All indices are block indices
    //The benchmark values
    double correct_mem_block_major[20] = { //j = 0, i = 0
                                           1,6,
                                           2,7,

                                           //j = 0, i = 1
                                           11,16,
                                           12,17,

                                           //j = 1, i = 0
                                           3,8,
                                           4,9,
                                           5,10,

                                           //j = 1,i = 1
                                           13,18,
                                           14,19,
                                           15,20};


    sparse_bispace<1> spb_1(4);
    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    spb_1.split(split_points);
    spb_2.split(split_points);

    sparse_bispace<2> two_d_input = spb_1 | spb_2;
    sparse_bispace<2> two_d_output = spb_2 | spb_1;
    sparse_btensor<2> bt(two_d_input,mem_row_major);

    sparse_btensor<2> bt_trans(two_d_output);
    letter i,j;
    bt_trans(i|j) = bt(j|i);

    sparse_btensor<2> correct(two_d_output,correct_mem_block_major,true);
    if(bt_trans != correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "labeled_btensor<N>::operator=(...) did not produce correct result");
    }
}

//This test loads both the input and the benchmark tensor from row major format, and thus is a good test
//of that capability as well
void sparse_btensor_test::test_permute_3d_row_major_210() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_permute_3d_row_major_210()";

    //Dimensions 3x4x5
    //ijk -> kji
    double mem_row_major[60] = { //i = 0
                                 1,2,3,4,5,//j = 0
                                 6,7,8,9,10,//j = 1
                                 11,12,13,14,15,//j = 2
                                 16,17,18,19,20,//j = 3

                                 //i = 1
                                 21,22,23,24,25,//j = 0
                                 26,27,28,29,30,//j = 1
                                 31,32,33,34,35,//j = 2
                                 36,37,38,39,40,//j = 3

                                 //i = 2
                                 41,42,43,44,45,//j = 0
                                 46,47,48,49,50,//j = 1
                                 51,52,53,54,55,//j = 2
                                 56,57,58,59,60};//j= 3

    double correct_row_major[60] = { //k = 0
                                     1,21,41,//j = 0
                                     6,26,46,//j = 1
                                     11,31,51,//j = 2
                                     16,36,56,//j = 3

                                     //k = 1
                                     2,22,42,//j = 0
                                     7,27,47,//j = 1
                                     12,32,52,//j = 2
                                     17,37,57,//j = 3

                                     //k = 2
                                     3,23,43,//j = 0
                                     8,28,48,//j = 1
                                     13,33,53,//j = 2
                                     18,38,58,//j = 3

                                     //k = 3
                                     4,24,44,
                                     9,29,49,
                                     14,34,54,
                                     19,39,59,
                                     
                                     //k = 4
                                     5,25,45, //j = 0
                                     10,30,50,//j = 1
                                     15,35,55,//j = 2
                                     20,40,60};//j = 3

    sparse_bispace<1> spb_1(3);
    sparse_bispace<1> spb_2(4);
    sparse_bispace<1> spb_3(5);

    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    std::vector<size_t> split_points_2;
    split_points_2.push_back(1);
    spb_2.split(split_points_2);

    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);


    sparse_bispace<3> three_d_input = spb_1 | spb_2 | spb_3;
    sparse_bispace<3> three_d_output = spb_3 | spb_2 | spb_1;
    sparse_btensor<3> bt(three_d_input,mem_row_major);

    sparse_btensor<3> bt_210(three_d_output);
    letter i,j,k;
    bt_210(k|j|i) = bt(i|j|k);

    sparse_btensor<3> correct(three_d_output,correct_row_major);
    if(bt_210 != correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "labeled_btensor<N>::operator=(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_permute_3d_block_major_120_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_permute_3d_block_major_120_sparse()";

    //3x4x5
    //Permutation is kij -> ijk 
	//Indices in comments are block indices
    double test_input_arr[35] = { //k = 0, i = 0; j = 0
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

    double correct_output_arr[35] = { //i = 0 j = 0 k = 0
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

    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    //Second bispace (mid index in input) and splitting
    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    spb_2.split(split_points_2);

    //Third bispace (fast index in input) and splitting
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


    sparse_btensor<3> bt(three_d_input,test_input_arr,true);

    sparse_btensor<3> bt_120(three_d_output);
    letter i,j,k;
    bt_120(i|j|k) = bt(k|i|j);

    sparse_btensor<3> correct(three_d_output,correct_output_arr,true);
    if(bt_120 != correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "labeled_btensor<N>::operator=(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_2d_2d()";

    //Both matrices stored slow->fast, so blas must transpose second one
    //dimensions: i = 4, k = 5, j = 6
    double test_input_arr_1[20] = {1,2,3,4,5,
                                   6,7,8,9,10,
                                   11,12,13,14,15,
                                   16,17,18,19,20};

    double test_input_arr_2[30] = {1,2,3,4,5,
                                   6,7,8,9,10,
                                   11,12,13,14,15,
                                   16,17,18,19,20,
                                   21,22,23,24,25,
                                   26,27,28,29,30};

    double correct_output_arr[24] = {55,130,205,280,355,430, 
                                     130,330,530,730,930,1130,
                                     205,530,855,1180,1505,1830,
                                     280,730,1180,1630,2080,2530};

    sparse_bispace<1> spb_i(4); 
    std::vector<size_t> split_points_i(1,2); 
    spb_i.split(split_points_i);

    sparse_bispace<1> spb_k(5); 
    std::vector<size_t> split_points_k(1,2); 
    spb_k.split(split_points_k);

    sparse_bispace<1> spb_j(6); 
    std::vector<size_t> split_points_j(1,2); 
    spb_j.split(split_points_j);


    sparse_btensor<2> bt_1(spb_i | spb_k,test_input_arr_1);
    sparse_btensor<2> bt_2(spb_j | spb_k,test_input_arr_2);

    sparse_bispace<2> spb_ij = spb_i | spb_j;
    sparse_btensor<2> result(spb_ij);
    letter i,j,k;
    result(i|j) = contract(k,bt_1(i|k),bt_2(j|k));

    sparse_btensor<2> correct_result(spb_ij,correct_output_arr);
    if(result != correct_result)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_2d_2d_sparse_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_2d_2d_sparse_dense()";

    //Both matrices stored slow->fast, so blas must transpose second one
    //dimensions: i = 4, k = 5, j = 6

    //Block major     
    double A_arr[20] = { //i = 0 //k = 0
                         1,2,3,
                         4,5,6,

                         //i = 1 //k = 0
                         7,8,9,
                         10,11,12};
                       
    //Row major
    double B_arr[30] = {1,2,3,4,5,
                        6,7,8,9,10,
                        11,12,13,14,15,
                        16,17,18,19,20,
                        21,22,23,24,25,
                        26,27,28,29,30};

    //Row major
    double C_correct_arr[24] = {14,44,74,104,134,164, 
                                32,107,182,257,332,407,
                                50,170,290,410,530,650,
                                68,233,398,563,728,893};


    sparse_bispace<1> spb_i(4); 
    std::vector<size_t> split_points_i(1,2); 
    spb_i.split(split_points_i);

    sparse_bispace<1> spb_k(5); 
    std::vector<size_t> split_points_k(1,3); 
    spb_k.split(split_points_k);

    sparse_bispace<1> spb_j(6); 
    std::vector<size_t> split_points_j(1,2); 
    spb_j.split(split_points_j);

    //Keep first block column of A, all of B
    std::vector< sequence<2,size_t> > sig_blocks_A(2); 
    sig_blocks_A[0][0] = 0;
    sig_blocks_A[0][1] = 0;
    sig_blocks_A[1][0] = 1;
    sig_blocks_A[1][1] = 0;

    sparse_btensor<2> A(spb_i % spb_k << sig_blocks_A,A_arr,true);
    sparse_btensor<2> B(spb_j | spb_k,B_arr);
    sparse_btensor<2> C(spb_i | spb_j);

    letter i,j,k;
    C(i|j) = contract(k,A(i|k),B(j|k));

    sparse_btensor<2> C_correct(spb_i | spb_j,C_correct_arr);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_3d_2d()";

    //Tensors are stored block-major for this test
    //dimensions: i = 3,j = 4, k = 5,l = 6
    //Contraction takes the form of A*B
    double test_input_arr_1[60] = {//i = 0 j = 0 k = 0 (1,2,2)
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


    double test_input_arr_2[30] = {//k = 0  l = 0
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

    double correct_output_arr[72] = {//i = 0 j = 0 l = 0 
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

    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    spb_k.split(split_points_k);

    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(3);
    spb_l.split(split_points_l);

    sparse_bispace<3> A_spb = spb_i | spb_j | spb_k;
    sparse_bispace<2> B_spb = spb_k | spb_l;
    sparse_bispace<3> C_spb = spb_i | spb_j | spb_l;


    sparse_btensor<3> A(A_spb,test_input_arr_1,true);
    sparse_btensor<2> B(B_spb,test_input_arr_2,true);
    sparse_btensor<3> C(C_spb);

    letter i,j,k,l;
    C(i|j|l) = contract(k,A(i|j|k),B(k|l));

    sparse_btensor<3> C_correct(C_spb,correct_output_arr,true);
    if(C != C_correct)
    {
        for(size_t i = 0; i < 72; ++i)
        {
            double one = C.get_data_ptr()[i]; 
            double two = C_correct.get_data_ptr()[i];
            if(one != two)
            {
                std::cout << "mine: " << one << "\n";
                std::cout << "correct: " << two << "\n";
            }
        }
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_3d_2d_sparse_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_3d_2d_sparse_dense()";

    //Both matrices stored slow->fast, so blas must transpose second one
    //dimensions: i = 3, j = 4, k = 5, l = 6

    //Block major     
    double A_arr[26] = { //i = 0 j = 0 //k = 1
                         1,
                         2,

                         //i = 0 //j = 1 //k = 0
                         3,4,
                         5,6,

                         //i = 0 //j = 1 //k = 2
                         7,8,
                         9,10,

                         //i = 1 //j = 0 //k = 1
                         11,
                         12,
                         13,
                         14,

                         //i = 1 //j = 1 //k = 1
                         15,
                         16,
                         17,
                         18,

                         //i = 1 //j = 1 //k = 2
                         19,20,
                         21,22,
                         23,24,
                         25,26};

                       
    //Row major
    double B_arr[30] = {1,2,3,4,5,
                        6,7,8,9,10,
                        11,12,13,14,15,
                        16,17,18,19,20,
                        21,22,23,24,25,
                        26,27,28,29,30};

    //Row major
    double C_correct_arr[72] = {//i = 0 
                                //j = 0
                                3,8,13,18,23,28, 
                                //j = 1
                                6,16,26,36,46,56,
                                //j = 2
                                79,189,299,409,519,629, 
                                //j = 3
                                103,253,403,553,703,853,

                                //i = 1
                                //j = 0
                                33,88,143,198,253,308, 
                                //j = 1
                                36,96,156,216,276,336,
                                //j = 2
                                221,491,761,1031,1301,1571,
                                //j = 3
                                242,537,832,1127,1422,1717,
                                
                                //i = 2
                                //j = 0
                                39,104,169,234,299,364, 
                                //j = 1
                                42,112,182,252,322,392,
                                //j = 2
                                263,583,903,1223,1543,1863, 
                                //j = 3
                                284,629,974,1319,1664,2009};

    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
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


    //Sparsity data
    std::vector< sequence<3,size_t> > sig_blocks_A(6); 
    sig_blocks_A[0][0] = 0;
    sig_blocks_A[0][1] = 0;
    sig_blocks_A[0][2] = 1;
    sig_blocks_A[1][0] = 0;
    sig_blocks_A[1][1] = 1;
    sig_blocks_A[1][2] = 0;
    sig_blocks_A[2][0] = 0;
    sig_blocks_A[2][1] = 1;
    sig_blocks_A[2][2] = 2;
    sig_blocks_A[3][0] = 1;
    sig_blocks_A[3][1] = 0;
    sig_blocks_A[3][2] = 1;
    sig_blocks_A[4][0] = 1;
    sig_blocks_A[4][1] = 1;
    sig_blocks_A[4][2] = 1;
    sig_blocks_A[5][0] = 1;
    sig_blocks_A[5][1] = 1;
    sig_blocks_A[5][2] = 2;

    sparse_bispace<3> spb_A = spb_i % spb_j % spb_k << sig_blocks_A;
    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<2> B(spb_l | spb_k,B_arr);
    sparse_bispace<3> spb_C = spb_A.contract(2) | spb_l;
    sparse_btensor<3> C(spb_C);

    letter i,j,k,l;
    C(i|j|l) = contract(k,A(i|j|k),B(l|k));

    sparse_btensor<3> C_correct(spb_C,C_correct_arr);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_3d_2d_sparse_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_3d_2d_sparse_dense()";

    //Multiplication of the form A*B, no transposes
    //dimensions: i = 3, j = 4, k = 5, l = 6

    //Block major     
    double A_arr[26] = { //i = 0 j = 0 k = 1
                         1,
                         2,

                         //i = 0 j = 1 k = 0
                         3,4,
                         5,6,

                         //i = 0 j = 1 k = 2
                         7,8,
                         9,10,

                         //i = 1 j = 0 k = 1
                         11,
                         12,
                         13,
                         14,

                         //i = 1 j = 1 k = 1
                         15,
                         16,
                         17,
                         18,

                         //i = 1 j = 1 k = 2
                         19,20,
                         21,22,
                         23,24,
                         25,26};

                       
    //Block major
    double B_arr[19] = { //k = 0 l = 1
                         1,2,3,
                         4,5,6,

                         //k = 0 l = 2
                         7,
                         8,

                         //k = 1 l = 2
                         9,

                         //k = 2 l = 0
                         10,11,
                         12,13,

                         //k = 2 l = 1
                         14,15,16,
                         17,18,19};

    //Block major
    double C_correct_arr[72] = {//i = 0 j = 0 l = 2
                                9,
                                18,

                                //i = 0 j = 1 l = 0
                                166,181,
                                210,229,
                                
                                //i = 0 j = 1 l = 1
                                253,275,297,
                                325,355,385,
                                
                                //i = 0 j = 1 l = 2
                                53,
                                83,

                                //i = 1 j = 0 l = 2 
                                99,
                                108,
                                117,
                                126,

                                //i = 1 j = 1 l = 0
                                430,469, 
                                474,517,
                                518,565,
                                562,613,

                                //i = 1 j = 1 l = 1
                                606,645,684,
                                668,711,754,
                                730,777,824,
                                792,843,894,
    
                                //i = 1 j = 1 l = 2
                                135,
                                144,
                                153,
                                162};

    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
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


    //Sparsity data
    std::vector< sequence<3,size_t> > sig_blocks_A(6); 
    sig_blocks_A[0][0] = 0;
    sig_blocks_A[0][1] = 0;
    sig_blocks_A[0][2] = 1;
    sig_blocks_A[1][0] = 0;
    sig_blocks_A[1][1] = 1;
    sig_blocks_A[1][2] = 0;
    sig_blocks_A[2][0] = 0;
    sig_blocks_A[2][1] = 1;
    sig_blocks_A[2][2] = 2;
    sig_blocks_A[3][0] = 1;
    sig_blocks_A[3][1] = 0;
    sig_blocks_A[3][2] = 1;
    sig_blocks_A[4][0] = 1;
    sig_blocks_A[4][1] = 1;
    sig_blocks_A[4][2] = 1;
    sig_blocks_A[5][0] = 1;
    sig_blocks_A[5][1] = 1;
    sig_blocks_A[5][2] = 2;

    std::vector< sequence<2,size_t> > sig_blocks_B(5); 
    sig_blocks_B[0][0] = 0;
    sig_blocks_B[0][1] = 1;
    sig_blocks_B[1][0] = 0;
    sig_blocks_B[1][1] = 2;
    sig_blocks_B[2][0] = 1;
    sig_blocks_B[2][1] = 2;
    sig_blocks_B[3][0] = 2;
    sig_blocks_B[3][1] = 0;
    sig_blocks_B[4][0] = 2;
    sig_blocks_B[4][1] = 1;

    sparse_bispace<3> spb_A = spb_i % spb_j % spb_k << sig_blocks_A;
    sparse_bispace<2> spb_B = spb_k % spb_l << sig_blocks_B;

    sparse_btensor<3> A(spb_A,A_arr,true);
    sparse_btensor<2> B(spb_B,B_arr,true);
    sparse_bispace<3> spb_C = spb_A.fuse(spb_B).contract(2);
    sparse_btensor<3> C(spb_C);

    letter i,j,k,l;
    C(i|j|l) = contract(k,A(i|j|k),B(k|l));

    sparse_btensor<3> C_correct(spb_C,C_correct_arr,true);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void sparse_btensor_test::test_contract2_two_indices_3d_2d_dense_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_contract2_two_indices_3d_2d_dense_dense()";
}

} // namespace libtensor
