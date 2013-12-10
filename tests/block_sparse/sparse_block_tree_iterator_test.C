#include <libtensor/block_sparse/sparse_block_tree_new.h>
#include "sparse_block_tree_iterator_test.h"

namespace libtensor {
    
void sparse_block_tree_iterator_test::perform() throw(libtest::test_exception)
{
    test_begin_3d();
    test_end_3d();
    test_incr_3d();
}

void sparse_block_tree_iterator_test::test_begin_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_iterator_test::test_begin_3d()";
    
    //Populate a branch node with a token entry
    size_t seq01_arr[3] = {1,2,3};
    
    std::vector< sequence<3,size_t> > block_tuples_list(1);
    for(size_t i = 0; i < 3; ++i) block_tuples_list[0][i] = seq01_arr[i];

    impl::sparse_block_tree_new<3> sbt(block_tuples_list);
    impl::sparse_block_tree_new<3>::iterator it = sbt.begin();

    //Check key
    for(size_t i = 0; i < 3; ++i)
    {
        if(it.key()[i] != block_tuples_list[0][i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_block_tree<3>::iterator::key() returned incorrect key");
        }
    }

    //Check that value can be set correctly
    std::vector<size_t> val(2,2);
    *it = val;
    for(size_t i = 0; i < 2; ++i)
    {
        if((*it)[i] != val[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_block_tree<3>::iterator::operator*() returned incorrect value");
        }
    }
}

//begin() should equal end() for an empty tree
void sparse_block_tree_iterator_test::test_end_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_iterator_test::test_end_3d()";
    
    std::vector< sequence<3,size_t> > block_tuples_list;

    impl::sparse_block_tree_new<3> sbt(block_tuples_list);
    impl::sparse_block_tree_new<3>::iterator beg = sbt.begin();
    impl::sparse_block_tree_new<3>::iterator end = sbt.end();

    if(beg != end)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree<3>::end() returned incorrect value");
    }
}

void sparse_block_tree_iterator_test::test_incr_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_iterator_test::test_incr_3d()";

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

    size_t m = 0;  
    for(impl::sparse_block_tree_new<3>::iterator sbt_it = sbt.begin(); sbt_it != sbt.end(); ++sbt_it)
    {
        std::vector<size_t> key = sbt_it.key();
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
    if(m != 21) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "iterator did not access all elements!");
    }
} 
    
} // namespace libtensor

