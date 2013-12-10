#include <libtensor/block_sparse/sparse_block_tree.h>
#include <libtensor/block_sparse/sparse_block_tree_new.h>
#include <libtensor/block_sparse/runtime_permutation.h>
#include "sparse_block_tree_test.h"

namespace libtensor {

void sparse_block_tree_test::perform() throw(libtest::test_exception)
{
    test_zero_order();
    test_unsorted_input();

    test_equality_false_2d();
    test_equality_true_2d();
    
    test_permute_3d();

    test_get_sub_key_block_list_invalid_key_size();
    test_get_sub_key_block_list_nonexistent_key();
    test_get_sub_key_block_list_2d();
    test_get_sub_key_block_list_3d();

    test_search_2d_invalid_key();
    test_search_3d();


    test_contract_3d_0();
    test_contract_3d_1();
    test_contract_3d_2();

    test_fuse_2d_2d();
    test_fuse_3d_2d();
    test_fuse_3d_3d_non_contiguous();
    test_fuse_3d_3d_multi_index();
}

//Cannot have a zero-order tree - throw an exception if this is requested
void sparse_block_tree_test::test_zero_order() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_zero_order()";
    
    //Must have only one key, otherwise we will trip the duplicate keys exception
    //and get a false positive
    std::vector< sequence<0,size_t> > block_tuples_list(1);
    
    bool threw_exception = false;
    try
    {
        impl::sparse_block_tree_new<0> sbt(block_tuples_list);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    
    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "Instantiating a zero-order tree did not cause an exception");
    }
}

void sparse_block_tree_test::test_equality_false_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_equality_false_2d()";
    
    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};
    
    std::vector< sequence<2,size_t> > block_tuples_list(7);
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];
    
    impl::sparse_block_tree_new<2> sbt_1(block_tuples_list);
    
    //Change one entry
    size_t seq7_arr_2[2] = {5,3};
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr_2[i];
    impl::sparse_block_tree_new<2> sbt_2(block_tuples_list);
    
    if(sbt_1 == sbt_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree<N>::operator==(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_equality_true_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_equality_true_2d()";
    
    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};
    
    std::vector< sequence<2,size_t> > block_tuples_list(7);
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];
    
    impl::sparse_block_tree_new<2> sbt_1(block_tuples_list);
    impl::sparse_block_tree_new<2> sbt_2(block_tuples_list);
    
    if(sbt_1 != sbt_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree<N>::operator==(...) returned incorrect value");
    }
}
    
void sparse_block_tree_test::test_unsorted_input() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_unsorted_input()";

    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {2,1}; //This one is invalid!
    size_t seq5_arr[2] = {2,5};

    std::vector< sequence<2,size_t> > block_tuples_list(5); 
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];

    bool threw_exception = false;
    try
    {
        impl::sparse_block_tree_new<2> sbt(block_tuples_list);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Unsorted input did not cause exception");
    }
}

void sparse_block_tree_test::test_get_sub_key_block_list_invalid_key_size() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_block_list_invalid_key_size()";

    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > block_tuples_list(7); 
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];

    sparse_block_tree<2> sbt(block_tuples_list);


    //First test key that is too long
    bool threw_exception = false;
    try
    {
        std::vector<size_t> too_big_key;
        too_big_key.push_back(5);
        too_big_key.push_back(2);
        sbt.get_sub_key_block_list(too_big_key);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "did not throw exception for key that is too large");
    }

    //Now test empty key
    threw_exception = false;
    try
    {
        sbt.get_sub_key_block_list(std::vector<size_t>());
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "did not throw exception for empty key");
    }

}

void sparse_block_tree_test::test_get_sub_key_block_list_nonexistent_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_block_list_nonexistent_key()";

    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > block_tuples_list(7); 
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];

    sparse_block_tree<2> sbt(block_tuples_list);


    bool threw_exception = false;
    try
    {
        std::vector<size_t> nonexistent_key(1,6);
        sbt.get_sub_key_block_list(nonexistent_key);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "did not throw exception for key does not exist");
    }
}

void sparse_block_tree_test::test_get_sub_key_block_list_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_block_list_2d()";

    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > block_tuples_list(7); 
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];

    sparse_block_tree<2> sbt(block_tuples_list);

    const block_list& bl = sbt.get_sub_key_block_list(std::vector<size_t>(1,4));


    if(bl[0] != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Dereferencing sub_key_block_list first time returned incorrect value");
    }
    if(bl[1] != 4)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Dereferencing sub_key_block_list second time returned incorrect value");
    }
} 

void sparse_block_tree_test::test_get_sub_key_block_list_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_block_list_3d()";

    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    sparse_block_tree<3> sbt(block_tuples_list);

    //Test a 1D key
    const block_list& bl = sbt.get_sub_key_block_list(std::vector<size_t>(1,2));
    std::vector<size_t> correct_vals_1d;
    correct_vals_1d.push_back(3);
    correct_vals_1d.push_back(4);
    correct_vals_1d.push_back(6);

    if(bl.size() != correct_vals_1d.size())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "1d key output is wrong size");
    }

    for(size_t i = 0; i < correct_vals_1d.size(); ++i)
    {
        if(bl[i] != correct_vals_1d[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sub_key_block_list returned incorrect value for 1d key");
        }
    }

    std::vector<size_t> key_2d;
    key_2d.push_back(2);
    key_2d.push_back(6);

    std::vector<size_t> correct_vals_2d;
    correct_vals_2d.push_back(3);
    correct_vals_2d.push_back(4);
    const block_list& bl2 = sbt.get_sub_key_block_list(key_2d);

    if(bl2.size() != correct_vals_2d.size())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "2d key output is wrong size");
    }
    for(size_t i = 0; i < correct_vals_2d.size(); ++i)
    {
        if(bl2[i] != correct_vals_2d[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sub_key_block_list returned incorrect value for 2d key");
        }
    }
} 

void sparse_block_tree_test::test_search_2d_invalid_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_search_2d_invalid_key()";

    size_t seq1_arr[2] = {1,2};
    size_t seq2_arr[2] = {1,5};
    size_t seq3_arr[2] = {2,3};
    size_t seq4_arr[2] = {4,1};
    size_t seq5_arr[2] = {4,4};
    size_t seq6_arr[2] = {5,1};
    size_t seq7_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > block_tuples_list(7); 
    for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];

    sparse_block_tree<2> sbt(block_tuples_list);

    sequence<7,size_t> val_seq;
    for(size_t i = 0; i < 7; ++i) val_seq[i] = i;
    size_t m = 0;  

    //Set all of the values in the tree
    for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = val_seq[m]; 
        ++m;
    }

    bool threw_exception = false;
    try
    {
        std::vector<size_t> key(2);
        key[0] = 1; 
        key[1] = 3; // (1,3) is not in tree
        sbt.search(key);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<2>::search(...) did not throw exception for invalid key");
    }
} 

void sparse_block_tree_test::test_search_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_search_3d()";

    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    sparse_block_tree<3> sbt(block_tuples_list);

    sequence<21,size_t> val_seq;
    for(size_t i = 0; i < 21; ++i) val_seq[i] = i;

    //Set all of the values in the tree
    size_t m = 0;  
    for(sparse_block_tree<3>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = val_seq[m]; 
        ++m;
    }

    std::vector<size_t> key(3);
    key[0] = 4; 
    key[1] = 1;
    key[2] = 4;
    if(sbt.search(key) != 9)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<3>::search(...) did not return correct value");
    }
} 

void sparse_block_tree_test::test_permute_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_permute_3d()";

    //Build the initial tree 
    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    impl::sparse_block_tree_new<3> sbt(block_tuples_list);

    //Set all of the values in the tree
    size_t m = 0;
    for(impl::sparse_block_tree_new<3>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = std::vector<size_t>(1,m); 
        ++m;
    }

    //permute first and last index
    runtime_permutation perm(3);
    perm.permute(0,2);

    impl::sparse_block_tree_new<3> permuted_sbt = sbt.permute(perm);

    //Build the benchmark tree
    size_t correct_seq01_arr[3] = {1,3,1};// orig pos: 2
    size_t correct_seq02_arr[3] = {1,3,2};// orig pos: 4
    size_t correct_seq03_arr[3] = {2,2,4};// orig pos:11
    size_t correct_seq04_arr[3] = {2,4,2};// orig pos: 5
    size_t correct_seq05_arr[3] = {3,2,1};// orig pos: 0 
    size_t correct_seq06_arr[3] = {3,6,2};// orig pos: 7
    size_t correct_seq07_arr[3] = {4,1,4};// orig pos: 9
    size_t correct_seq08_arr[3] = {4,1,5};// orig pos:15
    size_t correct_seq09_arr[3] = {4,6,2};// orig pos: 8
    size_t correct_seq10_arr[3] = {5,3,4};// orig pos:12
    size_t correct_seq11_arr[3] = {5,4,2};// orig pos: 6
    size_t correct_seq12_arr[3] = {5,4,7};// orig pos:18
    size_t correct_seq13_arr[3] = {6,2,5};// orig pos:16
    size_t correct_seq14_arr[3] = {6,3,4};// orig pos:13
    size_t correct_seq15_arr[3] = {6,4,7};// orig pos:19
    size_t correct_seq16_arr[3] = {7,1,4};// orig pos:10
    size_t correct_seq17_arr[3] = {7,2,1};// orig pos: 1
    size_t correct_seq18_arr[3] = {7,2,5};// orig pos:17
    size_t correct_seq19_arr[3] = {7,3,4};// orig pos:14
    size_t correct_seq20_arr[3] = {7,7,7};// orig pos:20
    size_t correct_seq21_arr[3] = {9,5,1};// orig pos: 3

    std::vector< sequence<3,size_t> > correct_block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[0][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[1][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[2][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[3][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[4][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[5][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[6][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[7][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[8][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[9][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[10][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[11][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[12][i] = correct_seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[13][i] = correct_seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[14][i] = correct_seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[15][i] = correct_seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[16][i] = correct_seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[17][i] = correct_seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[18][i] = correct_seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[19][i] = correct_seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_block_tuples_list[20][i] = correct_seq21_arr[i];

    
    impl::sparse_block_tree_new<3> correct_sbt(correct_block_tuples_list);

    //Set all of the values in the tree
    std::vector< std::vector<size_t> > correct_vals;
    correct_vals.push_back(std::vector<size_t>(1,2));
    correct_vals.push_back(std::vector<size_t>(1,4));
    correct_vals.push_back(std::vector<size_t>(1,11));
    correct_vals.push_back(std::vector<size_t>(1,5));
    correct_vals.push_back(std::vector<size_t>(1,0));
    correct_vals.push_back(std::vector<size_t>(1,7));
    correct_vals.push_back(std::vector<size_t>(1,9));
    correct_vals.push_back(std::vector<size_t>(1,15));
    correct_vals.push_back(std::vector<size_t>(1,8));
    correct_vals.push_back(std::vector<size_t>(1,12));
    correct_vals.push_back(std::vector<size_t>(1,6));
    correct_vals.push_back(std::vector<size_t>(1,18));
    correct_vals.push_back(std::vector<size_t>(1,16));
    correct_vals.push_back(std::vector<size_t>(1,13));
    correct_vals.push_back(std::vector<size_t>(1,19));
    correct_vals.push_back(std::vector<size_t>(1,10));
    correct_vals.push_back(std::vector<size_t>(1,1));
    correct_vals.push_back(std::vector<size_t>(1,17));
    correct_vals.push_back(std::vector<size_t>(1,14));
    correct_vals.push_back(std::vector<size_t>(1,20));
    correct_vals.push_back(std::vector<size_t>(1,3));

    size_t n = 0;
    for(impl::sparse_block_tree_new<3>::iterator sbt_it = correct_sbt.begin(); sbt_it != correct_sbt.end(); ++sbt_it)
    {
        *sbt_it = correct_vals[n];
        ++n;
    }

    //Ensure that the trees match
    impl::sparse_block_tree_new<3>::iterator my_it = permuted_sbt.begin();  
    impl::sparse_block_tree_new<3>::iterator correct_it = correct_sbt.begin();  
    for(size_t i = 0; i < 21; ++i)
    {
        const std::vector<size_t> my_it_key = my_it.key();
        const std::vector<size_t>  correct_it_key = correct_it.key();
        for(size_t seq_idx = 0; seq_idx < 3; ++seq_idx)
        {
            if(my_it_key[seq_idx] != correct_it_key[seq_idx])
            {
                fail_test(test_name,__FILE__,__LINE__,
                        "keys do not match");
            }
        }
        if(*my_it != *correct_it)
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "values do not match");
        }
        ++my_it;
        ++correct_it;
    }
}

void sparse_block_tree_test::test_contract_3d_0() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_0()";

    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    sparse_block_tree<3> sbt(block_tuples_list);
    sparse_block_tree<2> sbt_contracted = sbt.contract(0);

    //Build the correct tree
    size_t correct_seq01_arr[2] = {1,4};
    size_t correct_seq02_arr[2] = {1,7};
    size_t correct_seq03_arr[2] = {2,2};
    size_t correct_seq04_arr[2] = {2,3};
    size_t correct_seq05_arr[2] = {2,6};
    size_t correct_seq06_arr[2] = {2,7};
    size_t correct_seq07_arr[2] = {3,1};
    size_t correct_seq08_arr[2] = {3,5};
    size_t correct_seq09_arr[2] = {3,6};
    size_t correct_seq10_arr[2] = {3,7};
    size_t correct_seq11_arr[2] = {4,2};
    size_t correct_seq12_arr[2] = {4,5};
    size_t correct_seq13_arr[2] = {4,6};
    size_t correct_seq14_arr[2] = {5,9};
    size_t correct_seq15_arr[2] = {6,3};
    size_t correct_seq16_arr[2] = {6,4};
    size_t correct_seq17_arr[2] = {7,7};

    std::vector< sequence<2,size_t> > correct_block_tuples_list(17);
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[0][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[1][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[2][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[3][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[4][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[5][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[6][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[7][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[8][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[9][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[10][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[11][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[12][i] = correct_seq13_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[13][i] = correct_seq14_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[14][i] = correct_seq15_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[15][i] = correct_seq16_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[16][i] = correct_seq17_arr[i];

    sparse_block_tree<2> correct_sbt(correct_block_tuples_list);

    //Set both trees to have the same arbitrary values - don't care about these
    size_t m = 0;  
    sparse_block_tree<2>::iterator correct_sbt_it = correct_sbt.begin();
    for(sparse_block_tree<2>::iterator sbt_it = sbt_contracted.begin(); sbt_it != sbt_contracted.end(); ++sbt_it)
    {
        *sbt_it = m;
        *correct_sbt_it = m;
        ++correct_sbt_it;
        ++m;
    }

    if(sbt_contracted != correct_sbt)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::contract(...) returned incorrect value");
    }
} 

void sparse_block_tree_test::test_contract_3d_1() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_1()";

    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    sparse_block_tree<3> sbt(block_tuples_list);
    sparse_block_tree<2> sbt_contracted = sbt.contract(1);

    //Build the correct tree
    size_t correct_seq01_arr[2] = {1,1};
    size_t correct_seq02_arr[2] = {1,3};
    size_t correct_seq03_arr[2] = {1,7};
    size_t correct_seq04_arr[2] = {1,9};
    size_t correct_seq05_arr[2] = {2,1};
    size_t correct_seq06_arr[2] = {2,2};
    size_t correct_seq07_arr[2] = {2,3};
    size_t correct_seq08_arr[2] = {2,4};
    size_t correct_seq09_arr[2] = {2,5};
    size_t correct_seq10_arr[2] = {4,2};
    size_t correct_seq11_arr[2] = {4,4};
    size_t correct_seq12_arr[2] = {4,5};
    size_t correct_seq13_arr[2] = {4,6};
    size_t correct_seq14_arr[2] = {4,7};
    size_t correct_seq15_arr[2] = {5,4};
    size_t correct_seq16_arr[2] = {5,6};
    size_t correct_seq17_arr[2] = {5,7};
    size_t correct_seq18_arr[2] = {7,5};
    size_t correct_seq19_arr[2] = {7,6};
    size_t correct_seq20_arr[2] = {7,7};

    std::vector< sequence<2,size_t> > correct_block_tuples_list(20);
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[0][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[1][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[2][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[3][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[4][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[5][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[6][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[7][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[8][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[9][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[10][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[11][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[12][i] = correct_seq13_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[13][i] = correct_seq14_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[14][i] = correct_seq15_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[15][i] = correct_seq16_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[16][i] = correct_seq17_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[17][i] = correct_seq18_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[18][i] = correct_seq19_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[19][i] = correct_seq20_arr[i];

    sparse_block_tree<2> correct_sbt(correct_block_tuples_list);

    //Set both trees to have the same arbitrary values - don't care about these
    size_t m = 0;  
    sparse_block_tree<2>::iterator correct_sbt_it = correct_sbt.begin();
    for(sparse_block_tree<2>::iterator sbt_it = sbt_contracted.begin(); sbt_it != sbt_contracted.end(); ++sbt_it)
    {
        *sbt_it = m;
        *correct_sbt_it = m;
        ++correct_sbt_it;
        ++m;
    }

    if(sbt_contracted != correct_sbt)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::contract(...) returned incorrect value");
    }
} 

void sparse_block_tree_test::test_contract_3d_2() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_2()";

    size_t seq01_arr[3] = {1,2,3};
    size_t seq02_arr[3] = {1,2,7};
    size_t seq03_arr[3] = {1,3,1};
    size_t seq04_arr[3] = {1,5,9};
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

    std::vector< sequence<3,size_t> > block_tuples_list(21); 
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[1][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[2][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[3][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[4][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[5][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[6][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[7][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[8][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[9][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[10][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[11][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[12][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[13][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[14][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[15][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[16][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[17][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[18][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[19][i] = seq20_arr[i];
    for(size_t i = 0; i < 3; ++i) block_tuples_list[20][i] = seq21_arr[i];

    sparse_block_tree<3> sbt(block_tuples_list);
    sparse_block_tree<2> sbt_contracted = sbt.contract(2);

    //Build the correct tree
    size_t correct_seq01_arr[2] = {1,2};
    size_t correct_seq02_arr[2] = {1,3};
    size_t correct_seq03_arr[2] = {1,5};
    size_t correct_seq04_arr[2] = {2,3};
    size_t correct_seq05_arr[2] = {2,4};
    size_t correct_seq06_arr[2] = {2,6};
    size_t correct_seq07_arr[2] = {4,1};
    size_t correct_seq08_arr[2] = {4,2};
    size_t correct_seq09_arr[2] = {4,3};
    size_t correct_seq10_arr[2] = {5,1};
    size_t correct_seq11_arr[2] = {5,2};
    size_t correct_seq12_arr[2] = {7,4};
    size_t correct_seq13_arr[2] = {7,7};

    std::vector< sequence<2,size_t> > correct_block_tuples_list(13);
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[0][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[1][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[2][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[3][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[4][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[5][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[6][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[7][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[8][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[9][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[10][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[11][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[12][i] = correct_seq13_arr[i];

    sparse_block_tree<2> correct_sbt(correct_block_tuples_list);

    //Set both trees to have the same arbitrary values - don't care about these
    size_t m = 0;  
    sparse_block_tree<2>::iterator correct_sbt_it = correct_sbt.begin();
    for(sparse_block_tree<2>::iterator sbt_it = sbt_contracted.begin(); sbt_it != sbt_contracted.end(); ++sbt_it)
    {
        *sbt_it = m;
        *correct_sbt_it = m;
        ++correct_sbt_it;
        ++m;
    }

    if(sbt_contracted != correct_sbt)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::contract(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_fuse_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_2d_2d()";

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

    sparse_block_tree<2> sbt_1(sig_blocks_1);

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

    sparse_block_tree<2> sbt_2(sig_blocks_2);

    sparse_block_tree<3> sbt_fused = sbt_1.fuse(sbt_2); 

    //Correct tree
    size_t correct_seq00_arr[3] = {1,2,5};
    size_t correct_seq01_arr[3] = {1,2,9};
    size_t correct_seq02_arr[3] = {1,5,1};
    size_t correct_seq03_arr[3] = {1,5,5};
    size_t correct_seq04_arr[3] = {2,3,1};
    size_t correct_seq05_arr[3] = {4,1,2};
    size_t correct_seq06_arr[3] = {4,1,6};
    size_t correct_seq07_arr[3] = {4,4,3};
    size_t correct_seq08_arr[3] = {4,4,6};
    size_t correct_seq09_arr[3] = {5,1,2};
    size_t correct_seq10_arr[3] = {5,1,6};
    size_t correct_seq11_arr[3] = {5,2,5};
    size_t correct_seq12_arr[3] = {5,2,9};

    std::vector< sequence<3,size_t> > correct_sig_blocks(13);
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[0][i] = correct_seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[1][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[2][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[3][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[4][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[5][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[6][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[7][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[8][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[9][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[10][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[11][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[12][i] = correct_seq12_arr[i];

    sparse_block_tree<3> sbt_correct(correct_sig_blocks);
    if(sbt_fused != sbt_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::fuse(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_fuse_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_2d()";

    //Sparsity data 1
    size_t seq00_arr[3] = {1,2,3};
    size_t seq01_arr[3] = {1,2,7};
    size_t seq02_arr[3] = {1,3,1};
    size_t seq03_arr[3] = {1,5,9};
    size_t seq04_arr[3] = {2,3,1};
    size_t seq05_arr[3] = {2,4,2};
    size_t seq06_arr[3] = {2,4,5};
    size_t seq07_arr[3] = {2,6,3};
    size_t seq08_arr[3] = {2,6,4};
    size_t seq09_arr[3] = {4,1,4};
    size_t seq10_arr[3] = {4,1,7};
    size_t seq11_arr[3] = {4,2,2};
    size_t seq12_arr[3] = {4,3,5};
    size_t seq13_arr[3] = {4,3,6};
    size_t seq14_arr[3] = {4,3,7};
    size_t seq15_arr[3] = {5,1,4};
    size_t seq16_arr[3] = {5,2,6};
    size_t seq17_arr[3] = {5,2,7};
    size_t seq18_arr[3] = {7,4,5};
    size_t seq19_arr[3] = {7,4,6};
    size_t seq20_arr[3] = {7,7,7};

    std::vector< sequence<3,size_t> > sig_blocks_1(21); 
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[0][i] = seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[1][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[2][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[3][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[4][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[5][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[6][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[7][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[8][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[9][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[10][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[11][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[12][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[13][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[14][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[15][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[16][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[17][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[18][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[19][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[20][i] = seq20_arr[i];

    sparse_block_tree<3> sbt_1(sig_blocks_1);

    //Sparsity data 2
    size_t seq00_arr_2[2] = {1,2};
    size_t seq01_arr_2[2] = {1,6};
    size_t seq02_arr_2[2] = {2,5};
    size_t seq03_arr_2[2] = {2,9};
    size_t seq04_arr_2[2] = {3,1};
    size_t seq05_arr_2[2] = {4,3};
    size_t seq06_arr_2[2] = {4,6};
    size_t seq07_arr_2[2] = {5,1};
    size_t seq08_arr_2[2] = {5,5};
    size_t seq09_arr_2[2] = {6,2};
    size_t seq10_arr_2[2] = {6,7};
    size_t seq11_arr_2[2] = {7,4};
    size_t seq12_arr_2[2] = {7,6};

    std::vector< sequence<2,size_t> > sig_blocks_2(13); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[0][i] = seq00_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[1][i] = seq01_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[2][i] = seq02_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[3][i] = seq03_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[4][i] = seq04_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[5][i] = seq05_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[6][i] = seq06_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[7][i] = seq07_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[8][i] = seq08_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[9][i] = seq09_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[10][i] = seq10_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[11][i] = seq11_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[12][i] = seq12_arr_2[i];

    sparse_block_tree<2> sbt_2(sig_blocks_2);

    sparse_block_tree<4> sbt_fused = sbt_1.fuse(sbt_2); 

    //Correct tree
    size_t correct_seq00_arr[4] = {1,2,3,1};
    size_t correct_seq01_arr[4] = {1,2,7,4};
    size_t correct_seq02_arr[4] = {1,2,7,6};
    size_t correct_seq03_arr[4] = {1,3,1,2};
    size_t correct_seq04_arr[4] = {1,3,1,6};
    size_t correct_seq05_arr[4] = {2,3,1,2};
    size_t correct_seq06_arr[4] = {2,3,1,6};
    size_t correct_seq07_arr[4] = {2,4,2,5};
    size_t correct_seq08_arr[4] = {2,4,2,9};
    size_t correct_seq09_arr[4] = {2,4,5,1};
    size_t correct_seq10_arr[4] = {2,4,5,5};
    size_t correct_seq11_arr[4] = {2,6,3,1};
    size_t correct_seq12_arr[4] = {2,6,4,3};
    size_t correct_seq13_arr[4] = {2,6,4,6};
    size_t correct_seq14_arr[4] = {4,1,4,3};
    size_t correct_seq15_arr[4] = {4,1,4,6};
    size_t correct_seq16_arr[4] = {4,1,7,4};
    size_t correct_seq17_arr[4] = {4,1,7,6};
    size_t correct_seq18_arr[4] = {4,2,2,5};
    size_t correct_seq19_arr[4] = {4,2,2,9};
    size_t correct_seq20_arr[4] = {4,3,5,1};
    size_t correct_seq21_arr[4] = {4,3,5,5};
    size_t correct_seq22_arr[4] = {4,3,6,2};
    size_t correct_seq23_arr[4] = {4,3,6,7};
    size_t correct_seq24_arr[4] = {4,3,7,4};
    size_t correct_seq25_arr[4] = {4,3,7,6};
    size_t correct_seq26_arr[4] = {5,1,4,3};
    size_t correct_seq27_arr[4] = {5,1,4,6};
    size_t correct_seq28_arr[4] = {5,2,6,2};
    size_t correct_seq29_arr[4] = {5,2,6,7};
    size_t correct_seq30_arr[4] = {5,2,7,4};
    size_t correct_seq31_arr[4] = {5,2,7,6};
    size_t correct_seq32_arr[4] = {7,4,5,1};
    size_t correct_seq33_arr[4] = {7,4,5,5};
    size_t correct_seq34_arr[4] = {7,4,6,2};
    size_t correct_seq35_arr[4] = {7,4,6,7};
    size_t correct_seq36_arr[4] = {7,7,7,4};
    size_t correct_seq37_arr[4] = {7,7,7,6};

    std::vector< sequence<4,size_t> > correct_sig_blocks(38);
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[0][i] = correct_seq00_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[1][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[2][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[3][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[4][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[5][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[6][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[7][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[8][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[9][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[10][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[11][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[12][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[13][i] = correct_seq13_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[14][i] = correct_seq14_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[15][i] = correct_seq15_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[16][i] = correct_seq16_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[17][i] = correct_seq17_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[18][i] = correct_seq18_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[19][i] = correct_seq19_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[20][i] = correct_seq20_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[21][i] = correct_seq21_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[22][i] = correct_seq22_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[23][i] = correct_seq23_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[24][i] = correct_seq24_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[25][i] = correct_seq25_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[26][i] = correct_seq26_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[27][i] = correct_seq27_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[28][i] = correct_seq28_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[29][i] = correct_seq29_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[30][i] = correct_seq30_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[31][i] = correct_seq31_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[32][i] = correct_seq32_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[33][i] = correct_seq33_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[34][i] = correct_seq34_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[35][i] = correct_seq35_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[36][i] = correct_seq36_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[37][i] = correct_seq37_arr[i];

    sparse_block_tree<4> sbt_correct(correct_sig_blocks);

    //Set them all to the same dummy values - dont' care about values for this test
    size_t m = 0; 
    sparse_block_tree<4>::iterator it = sbt_fused.begin();
    for(sparse_block_tree<4>::iterator correct_it = sbt_correct.begin(); correct_it != sbt_correct.end(); ++correct_it)
    {
        *it = m; 
        *correct_it = m;
        ++it;
        ++m;
    }

    if(sbt_fused != sbt_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::fuse(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_fuse_3d_3d_non_contiguous() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_3d_non_contiguous()";

    //Sparsity data 1
    size_t seq00_arr[3] = {1,2,3};
    size_t seq01_arr[3] = {1,2,7};
    size_t seq02_arr[3] = {1,3,1};
    size_t seq03_arr[3] = {1,5,9};
    size_t seq04_arr[3] = {2,3,1};
    size_t seq05_arr[3] = {2,4,2};
    size_t seq06_arr[3] = {2,4,5};
    size_t seq07_arr[3] = {2,6,3};
    size_t seq08_arr[3] = {2,6,4};

    std::vector< sequence<3,size_t> > sig_blocks_1(9); 
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[0][i] = seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[1][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[2][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[3][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[4][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[5][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[6][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[7][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[8][i] = seq08_arr[i];

    sparse_block_tree<3> sbt_1(sig_blocks_1);

    //Sparsity data 2
    size_t seq00_arr_2[3] = {1,3,1};
    size_t seq01_arr_2[3] = {1,3,5};
    size_t seq02_arr_2[3] = {1,3,7};
    size_t seq03_arr_2[3] = {1,6,2};
    size_t seq04_arr_2[3] = {1,6,3};
    size_t seq05_arr_2[3] = {1,6,5};
    size_t seq06_arr_2[3] = {1,7,1};
    size_t seq07_arr_2[3] = {2,4,4};
    size_t seq08_arr_2[3] = {2,5,2};
    size_t seq09_arr_2[3] = {2,5,3};
    size_t seq10_arr_2[3] = {3,2,9};
    size_t seq11_arr_2[3] = {3,4,8};
    size_t seq12_arr_2[3] = {4,1,2};
    size_t seq13_arr_2[3] = {4,8,1};
    size_t seq14_arr_2[3] = {4,9,4};
    size_t seq15_arr_2[3] = {5,7,1};
    size_t seq16_arr_2[3] = {6,2,4};
    size_t seq17_arr_2[3] = {6,2,5};
    size_t seq18_arr_2[3] = {6,3,2};
    size_t seq19_arr_2[3] = {7,1,8};
    size_t seq20_arr_2[3] = {7,8,2};



    std::vector< sequence<3,size_t> > sig_blocks_2(21); 
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[0][i] = seq00_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[1][i] = seq01_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[2][i] = seq02_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[3][i] = seq03_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[4][i] = seq04_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[5][i] = seq05_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[6][i] = seq06_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[7][i] = seq07_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[8][i] = seq08_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[9][i] = seq09_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[10][i] = seq10_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[11][i] = seq11_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[12][i] = seq12_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[13][i] = seq13_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[14][i] = seq14_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[15][i] = seq15_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[16][i] = seq16_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[17][i] = seq17_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[18][i] = seq18_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[19][i] = seq19_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[20][i] = seq20_arr_2[i];

    sparse_block_tree<3> sbt_2(sig_blocks_2);


    sparse_block_tree<5> sbt_fused = sbt_1.fuse(sbt_2,sequence<1,size_t>(2),sequence<1,size_t>(1)); 

    size_t correct_seq00_arr[5] = {1,2,3,1,1};
    size_t correct_seq01_arr[5] = {1,2,3,1,5};
    size_t correct_seq02_arr[5] = {1,2,3,1,7};
    size_t correct_seq03_arr[5] = {1,2,3,6,2};
    size_t correct_seq04_arr[5] = {1,2,7,1,1};
    size_t correct_seq05_arr[5] = {1,2,7,5,1};
    size_t correct_seq06_arr[5] = {1,3,1,4,2};
    size_t correct_seq07_arr[5] = {1,3,1,7,8};
    size_t correct_seq08_arr[5] = {1,5,9,4,4};
    size_t correct_seq09_arr[5] = {2,3,1,4,2};
    size_t correct_seq10_arr[5] = {2,3,1,7,8};
    size_t correct_seq11_arr[5] = {2,4,2,3,9};
    size_t correct_seq12_arr[5] = {2,4,2,6,4};
    size_t correct_seq13_arr[5] = {2,4,2,6,5};
    size_t correct_seq14_arr[5] = {2,4,5,2,2};
    size_t correct_seq15_arr[5] = {2,4,5,2,3};
    size_t correct_seq16_arr[5] = {2,6,3,1,1};
    size_t correct_seq17_arr[5] = {2,6,3,1,5};
    size_t correct_seq18_arr[5] = {2,6,3,1,7};
    size_t correct_seq19_arr[5] = {2,6,3,6,2};
    size_t correct_seq20_arr[5] = {2,6,4,2,4};
    size_t correct_seq21_arr[5] = {2,6,4,3,8};

    std::vector< sequence<5,size_t> > correct_sig_blocks(22);
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[0][i] = correct_seq00_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[1][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[2][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[3][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[4][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[5][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[6][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[7][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[8][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[9][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[10][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[11][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[12][i] = correct_seq12_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[13][i] = correct_seq13_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[14][i] = correct_seq14_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[15][i] = correct_seq15_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[16][i] = correct_seq16_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[17][i] = correct_seq17_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[18][i] = correct_seq18_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[19][i] = correct_seq19_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[20][i] = correct_seq20_arr[i];
    for(size_t i = 0; i < 5; ++i) correct_sig_blocks[21][i] = correct_seq21_arr[i];

    sparse_block_tree<5> sbt_correct(correct_sig_blocks);

    //Set them all to the same dummy values - dont' care about values for this test
    size_t m = 0; 
    sparse_block_tree<5>::iterator it = sbt_fused.begin();
    for(sparse_block_tree<5>::iterator correct_it = sbt_correct.begin(); correct_it != sbt_correct.end(); ++correct_it)
    {
        *it = m; 
        *correct_it = m;
        ++it;
        ++m;
    }
    if(sbt_fused != sbt_correct)
    {
        std::cout << "CORRECT:\n";
        for(sparse_block_tree<5>::iterator it = sbt_correct.begin(); it != sbt_correct.end(); ++it)
        {
            std::cout << "(";
            for(size_t j = 0; j < 5; ++j)
            {
                std::cout << it.key()[j] << ",";
            }
            std::cout << "): " << *it << "\n";
        }
        std::cout << "--------------------\n";
        std::cout << "MINE:\n";
        for(sparse_block_tree<5>::iterator it = sbt_fused.begin(); it != sbt_fused.end(); ++it)
        {
            std::cout << "(";
            for(size_t j = 0; j < 5; ++j)
            {
                std::cout << it.key()[j] << ",";
            }
            std::cout << "): " << *it << "\n";
        }
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::fuse(...) returned incorrect value");
    }
}

//ijk fused to jkl 
void sparse_block_tree_test::test_fuse_3d_3d_multi_index() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_3d_multi_index";
    //Sparsity data 1
    size_t seq00_arr[3] = {1,2,3};
    size_t seq01_arr[3] = {1,2,7};
    size_t seq02_arr[3] = {1,3,1};
    size_t seq03_arr[3] = {1,5,9};
    size_t seq04_arr[3] = {2,3,1};
    size_t seq05_arr[3] = {2,4,2};
    size_t seq06_arr[3] = {2,4,5};
    size_t seq07_arr[3] = {2,6,3};
    size_t seq08_arr[3] = {2,6,4};
    size_t seq09_arr[3] = {4,1,4};
    size_t seq10_arr[3] = {4,1,7};
    size_t seq11_arr[3] = {4,2,2};
    size_t seq12_arr[3] = {4,3,5};
    size_t seq13_arr[3] = {4,3,6};
    size_t seq14_arr[3] = {4,3,7};
    size_t seq15_arr[3] = {5,1,4};
    size_t seq16_arr[3] = {5,2,6};
    size_t seq17_arr[3] = {5,2,7};
    size_t seq18_arr[3] = {7,4,5};
    size_t seq19_arr[3] = {7,4,6};
    size_t seq20_arr[3] = {7,7,7};

    std::vector< sequence<3,size_t> > sig_blocks_1(21); 
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[0][i] = seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[1][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[2][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[3][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[4][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[5][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[6][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[7][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[8][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[9][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[10][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[11][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[12][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[13][i] = seq13_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[14][i] = seq14_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[15][i] = seq15_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[16][i] = seq16_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[17][i] = seq17_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[18][i] = seq18_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[19][i] = seq19_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_1[20][i] = seq20_arr[i];

    sparse_block_tree<3> sbt_1(sig_blocks_1);

    //Sparsity data 2
    size_t seq00_arr_2[3] = {1,3,1};
    size_t seq01_arr_2[3] = {1,3,5};
    size_t seq02_arr_2[3] = {1,3,7};
    size_t seq03_arr_2[3] = {1,6,2};
    size_t seq04_arr_2[3] = {1,6,3};
    size_t seq05_arr_2[3] = {1,6,5};
    size_t seq06_arr_2[3] = {1,7,1};
    size_t seq07_arr_2[3] = {2,4,4};
    size_t seq08_arr_2[3] = {2,5,2};
    size_t seq09_arr_2[3] = {2,5,3};
    size_t seq10_arr_2[3] = {3,2,9};
    size_t seq11_arr_2[3] = {3,4,8};
    size_t seq12_arr_2[3] = {4,1,2};
    size_t seq13_arr_2[3] = {4,8,1};
    size_t seq14_arr_2[3] = {4,9,4};
    size_t seq15_arr_2[3] = {5,7,1};
    size_t seq16_arr_2[3] = {6,2,4};
    size_t seq17_arr_2[3] = {6,2,5};
    size_t seq18_arr_2[3] = {6,3,2};
    size_t seq19_arr_2[3] = {7,1,8};
    size_t seq20_arr_2[3] = {7,8,2};



    std::vector< sequence<3,size_t> > sig_blocks_2(21); 
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[0][i] = seq00_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[1][i] = seq01_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[2][i] = seq02_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[3][i] = seq03_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[4][i] = seq04_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[5][i] = seq05_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[6][i] = seq06_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[7][i] = seq07_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[8][i] = seq08_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[9][i] = seq09_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[10][i] = seq10_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[11][i] = seq11_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[12][i] = seq12_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[13][i] = seq13_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[14][i] = seq14_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[15][i] = seq15_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[16][i] = seq16_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[17][i] = seq17_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[18][i] = seq18_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[19][i] = seq19_arr_2[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks_2[20][i] = seq20_arr_2[i];

    sparse_block_tree<3> sbt_2(sig_blocks_2);


    sequence<2,size_t> lhs_fuse_points;
    lhs_fuse_points[0] = 1;
    lhs_fuse_points[1] = 2;
    sequence<2,size_t> rhs_fuse_points;
    rhs_fuse_points[0] = 0;
    rhs_fuse_points[1] = 1;
    sparse_block_tree<4> sbt_fused = sbt_1.fuse(sbt_2,lhs_fuse_points,rhs_fuse_points); 

    //Correct answer
    //Coupling sparsity creates a VERY SPARSE structure
    size_t correct_seq00_arr[4] = {2,6,3,2};
    size_t correct_seq01_arr[4] = {4,1,7,1};

    std::vector< sequence<4,size_t> > correct_sig_blocks(2); 
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[0][i] = correct_seq00_arr[i];
    for(size_t i = 0; i < 4; ++i) correct_sig_blocks[1][i] = correct_seq01_arr[i];
    
    sparse_block_tree<4> sbt_correct(correct_sig_blocks);

    //Set them all to the same dummy values - dont' care about values for this test
    size_t m = 0; 
    sparse_block_tree<4>::iterator it = sbt_fused.begin();
    for(sparse_block_tree<4>::iterator correct_it = sbt_correct.begin(); correct_it != sbt_correct.end(); ++correct_it)
    {
        *it = m; 
        *correct_it = m;
        ++it;
        ++m;
    }
    if(sbt_fused != sbt_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>L::fuse(...) returned incorrect value");
    }
}

} // namespace libtensor

