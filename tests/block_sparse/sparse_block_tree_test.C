#include <libtensor/block_sparse/sparse_block_tree.h>
#include "sparse_block_tree_test.h"


//TODO REMOVE
#include <iostream>

namespace libtensor { 

void sparse_block_tree_test::perform() throw(libtest::test_exception)
{
    test_unsorted_input();

    test_get_sub_key_iterator_invalid_key_size();
    test_get_sub_key_iterator_nonexistent_key();
    test_get_sub_key_iterator_2d();

    test_iterator_2d_key();
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
        std::cout << "value: " << *sk_it << "\n";
        fail_test(test_name,__FILE__,__LINE__,
                "Dereferencing sub_key_iterator second time returned incorrect value");
    }
} 

void sparse_block_tree_test::test_iterator_2d_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_iterator_2d_deref()";

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
                    "Dereferencing iterator returned incorrect key value");
        }
    }
} 

/*void sparse_block_tree_test::test_iterator_2d() throw(libtest::test_exception)*/
/*{*/
    /*static const char *test_name = "sparse_block_tree_test::test_get_sub_key_iterator()";*/

    /*size_t seq1_arr[2] = {1,2};*/
    /*size_t seq2_arr[2] = {1,5};*/
    /*size_t seq3_arr[2] = {2,3};*/
    /*size_t seq4_arr[2] = {4,1};*/
    /*size_t seq5_arr[2] = {4,4};*/
    /*size_t seq6_arr[2] = {5,1};*/
    /*size_t seq7_arr[2] = {5,2};*/

    /*std::vector< sequence<2,size_t> > block_tuples_list(7); */
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq1_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq2_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq3_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq4_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[4][i] = seq5_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[5][i] = seq6_arr[i];*/
    /*for(size_t i = 0; i < 2; ++i) block_tuples_list[6][i] = seq7_arr[i];*/

    /*sparse_block_tree<2> sbt(block_tuples_list);*/

    /*for(sparse_block_tree<2>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)*/
    /*{*/
    /*}*/
/*} */

} // namespace libtensor

