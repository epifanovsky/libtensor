
#include <libtensor/block_sparse/sparse_btensor.h>
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
    static const char *test_name = "sparse_btensor_test::test_permute_2d_block_major()";

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

} // namespace libtensor
