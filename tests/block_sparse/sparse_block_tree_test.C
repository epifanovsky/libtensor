#include <libtensor/block_sparse/sparse_block_tree.h>
#include <libtensor/core/permutation.h>
#include "sparse_block_tree_test.h"

namespace libtensor { 

void sparse_block_tree_test::perform() throw(libtest::test_exception)
{
    test_unsorted_input();

    test_get_sub_key_iterator_invalid_key_size();
    test_get_sub_key_iterator_nonexistent_key();
    test_get_sub_key_iterator_2d();

    test_iterator_2d_key();
    test_iterator_2d_incr();
    test_iterator_2d_set();
    test_iterator_3d_incr();

    test_search_2d_invalid_key();
    test_search_2d();
    test_search_3d();

    test_permute_2d();
    test_permute_3d();

    test_equality_false_2d();
    test_equality_true_2d();
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
        sparse_block_tree<2> sbt(block_tuples_list);
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

void sparse_block_tree_test::test_get_sub_key_iterator_invalid_key_size() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_iterator_invalid_key_size()";

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
        sparse_block_tree<2>::sub_key_iterator sk_it = sbt.get_sub_key_iterator(too_big_key);
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
        sparse_block_tree<2>::sub_key_iterator sk_it = sbt.get_sub_key_iterator(std::vector<size_t>());
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

void sparse_block_tree_test::test_get_sub_key_iterator_nonexistent_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_iterator_nonexistent_key()";

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
        sparse_block_tree<2>::sub_key_iterator sk_it = sbt.get_sub_key_iterator(nonexistent_key);
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

void sparse_block_tree_test::test_get_sub_key_iterator_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_key_iterator()";

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

    sparse_block_tree<2>::sub_key_iterator sk_it = sbt.get_sub_key_iterator(std::vector<size_t>(1,4));


    if(*sk_it != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Dereferencing sub_key_iterator first time returned incorrect value");
    }
    ++sk_it; 
    if(*sk_it != 4)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Dereferencing sub_key_iterator second time returned incorrect value");
    }
} 

void sparse_block_tree_test::test_iterator_2d_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_iterator_2d_key()";

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

    sparse_block_tree<2>::iterator sbt_it = sbt.begin();

    sequence<2,size_t> key = sbt_it.key();
    for(size_t i = 0; i < 2; ++i)
    {
        if(key[i] != block_tuples_list[0][i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "iterator::key() returned incorrect key value");
        }
    }
} 

void sparse_block_tree_test::test_iterator_2d_incr() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_iterator_2d_incr()";

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

    size_t m = 0;  
    for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        const sequence<2,size_t>& key = sbt_it.key();
        for(size_t i = 0; i < 2; ++i)
        {
            if(key[i] != block_tuples_list[m][i])
            {
                fail_test(test_name,__FILE__,__LINE__,
                        "iterator::key() returned incorrect key value");
            }
        }
        ++m;
    }
} 

void sparse_block_tree_test::test_iterator_2d_set() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_iterator_2d_set()";

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

    //Check all of the set values
    m = 0;
    for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = val_seq[m]; 
        if(*sbt_it != val_seq[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "Dereferencing iterator returned incorrect value");
        }
        ++m;
    }
} 

void sparse_block_tree_test::test_iterator_3d_incr() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_iterator_3d_incr()";

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

    size_t m = 0;  
    for(sparse_block_tree<3>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        const sequence<3,size_t>& key = sbt_it.key();
        for(size_t i = 0; i < 2; ++i)
        {
            if(key[i] != block_tuples_list[m][i])
            {
                fail_test(test_name,__FILE__,__LINE__,
                        "iterator::key() returned incorrect key value");
            }
        }
        ++m;
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

void sparse_block_tree_test::test_search_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_search_2d()";

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

    //Set all of the values in the tree
    size_t m = 0;  
    for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = val_seq[m]; 
        ++m;
    }

    std::vector<size_t> key(2);
    key[0] = 4; 
    key[1] = 1;
    sparse_block_tree<2>::iterator sbt_it = sbt.search(key);
    if(*sbt_it != 3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<2>::search(...) did not return correct value");
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
    sparse_block_tree<3>::iterator sbt_it = sbt.search(key);
    if(*sbt_it != 9)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<3>::search(...) did not return correct value");
    }
} 

void sparse_block_tree_test::test_permute_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_permute_2d()";

    //Build the initial tree
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

    //Set all of the values in the tree
    size_t m = 0;
    for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = m; 
        ++m;
    }

    //Permute the tree
    permutation<2> perm;
    perm.permute(0,1);
    sparse_block_tree<2> permuted_sbt = sbt.permute(perm);


    //Build the benchmark tree
    size_t correct_seq1_arr[2] = {1,4};// orig pos: 3
    size_t correct_seq2_arr[2] = {1,5};// orig pos: 5
    size_t correct_seq3_arr[2] = {2,1};// orig pos: 0
    size_t correct_seq4_arr[2] = {2,5};// orig pos: 6
    size_t correct_seq5_arr[2] = {3,2};// orig pos: 2
    size_t correct_seq6_arr[2] = {4,4};// orig pos: 4
    size_t correct_seq7_arr[2] = {5,1};// orig pos: 1
    
    std::vector<size_t> correct_vals;
    correct_vals.push_back(3);
    correct_vals.push_back(5);
    correct_vals.push_back(0);
    correct_vals.push_back(6);
    correct_vals.push_back(2);
    correct_vals.push_back(4);
    correct_vals.push_back(1);

    std::vector< sequence<2,size_t> > correct_block_tuples_list(7); 
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[0][i] = correct_seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[1][i] = correct_seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[2][i] = correct_seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[3][i] = correct_seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[4][i] = correct_seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[5][i] = correct_seq6_arr[i];
    for(size_t i = 0; i < 2; ++i) correct_block_tuples_list[6][i] = correct_seq7_arr[i];

    sparse_block_tree<2> correct_sbt(correct_block_tuples_list);
    size_t n = 0; 
    for(sparse_block_tree<2>::iterator sbt_it = correct_sbt.begin(); sbt_it != correct_sbt.end(); ++sbt_it)
    {
        *sbt_it = correct_vals[n]; 
        ++n;
    }

    //Ensure that the trees match
    sparse_block_tree<2>::iterator my_it = permuted_sbt.begin();  
    sparse_block_tree<2>::iterator correct_it = correct_sbt.begin();  
    for(size_t i = 0; i < 7; ++i)
    {
        const sequence<2,size_t> my_it_key = my_it.key();
        const sequence<2,size_t> correct_it_key = correct_it.key();
        for(size_t seq_idx = 0; seq_idx < 2; ++seq_idx)
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

    sparse_block_tree<3> sbt(block_tuples_list);

    //Set all of the values in the tree
    size_t m = 0;
    for(sparse_block_tree<3>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        *sbt_it = m; 
        ++m;
    }

    //permute first and last index
    permutation<3> perm;
    perm.permute(0,2);

    sparse_block_tree<3> permuted_sbt = sbt.permute(perm);

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

    
    sparse_block_tree<3> correct_sbt(correct_block_tuples_list);

    //Set all of the values in the tree
    std::vector<size_t> correct_vals;
    correct_vals.push_back(2);
    correct_vals.push_back(4);
    correct_vals.push_back(11);
    correct_vals.push_back(5);
    correct_vals.push_back(0);
    correct_vals.push_back(7);
    correct_vals.push_back(9);
    correct_vals.push_back(15);
    correct_vals.push_back(8);
    correct_vals.push_back(12);
    correct_vals.push_back(6);
    correct_vals.push_back(18);
    correct_vals.push_back(16);
    correct_vals.push_back(13);
    correct_vals.push_back(19);
    correct_vals.push_back(10);
    correct_vals.push_back(1);
    correct_vals.push_back(17);
    correct_vals.push_back(14);
    correct_vals.push_back(20);
    correct_vals.push_back(3);

    
    size_t n = 0;
    for(sparse_block_tree<3>::iterator sbt_it = correct_sbt.begin(); sbt_it != correct_sbt.end(); ++sbt_it)
    {
        *sbt_it = correct_vals[n];
        ++n;
    }

    //Ensure that the trees match
    sparse_block_tree<3>::iterator my_it = permuted_sbt.begin();  
    sparse_block_tree<3>::iterator correct_it = correct_sbt.begin();  
    for(size_t i = 0; i < 21; ++i)
    {
        const sequence<3,size_t> my_it_key = my_it.key();
        const sequence<3,size_t> correct_it_key = correct_it.key();
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

    sparse_block_tree<2> sbt_1(block_tuples_list);
    
    //Change one entry
    size_t seq7_arr_2[2] = {5,3};
    for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr_2[i];
    sparse_block_tree<2> sbt_2(block_tuples_list);

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

    sparse_block_tree<2> sbt_1(block_tuples_list);
    sparse_block_tree<2> sbt_2(block_tuples_list);

    if(sbt_1 != sbt_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<N>::operator==(...) returned incorrect value");
    }
} 

} // namespace libtensor

