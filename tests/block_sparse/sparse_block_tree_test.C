#include <libtensor/block_sparse/sparse_block_tree.h>
#include <libtensor/block_sparse/sparse_bispace.h>
#include <libtensor/block_sparse/runtime_permutation.h>
#include "sparse_block_tree_test.h"

using namespace std;

namespace libtensor {

//Test fixtures
namespace {

class two_d_test_f {
    static std::vector< sequence<2,size_t> > init_keys() 
    {
        //All 2d blocks are size 4
        size_t seq1_arr[2] = {1,2}; //offset 0
        size_t seq2_arr[2] = {1,5}; //offset 4
        size_t seq3_arr[2] = {2,3}; //offset 8
        size_t seq4_arr[2] = {4,1}; //offset 12
        size_t seq5_arr[2] = {4,4}; //offset 16
        size_t seq6_arr[2] = {5,1}; //offset 20
        size_t seq7_arr[2] = {5,2}; //offset 24
        size_t seq8_arr[2] = {7,4}; //offset 28
        
        std::vector< sequence<2,size_t> > key_list(8);
        for(size_t i = 0; i < 2; ++i) key_list[0][i] = seq1_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[1][i] = seq2_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[2][i] = seq3_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[3][i] = seq4_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[4][i] = seq5_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[5][i] = seq6_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[6][i] = seq7_arr[i];
        for(size_t i = 0; i < 2; ++i) key_list[7][i] = seq8_arr[i];
        return key_list;
    }

    static std::vector< sparse_bispace<1> > init_subspaces()
    {
        sparse_bispace<1> spb_1(16);
        vector<size_t> split_points_1;
        for(size_t i = 2; i < 16; i += 2)
        {
            split_points_1.push_back(i);
        }
        spb_1.split(split_points_1);

        return std::vector< sparse_bispace<1> >(2,spb_1);
    }

    static std::vector< sparse_block_tree::value_t > init_values()
    {
        size_t offset_arr[8] = {0,4,8,12,16,20,24,28};
        size_t size_arr[8] = {4,4,4,4,4,4,4,4};
        vector< sparse_block_tree::value_t > values;
        for(size_t i = 0; i < 8; ++i)
        {
            values.push_back(sparse_block_tree::value_t(1,off_dim_pair(offset_arr[i],size_arr[i])));
        }
        return values;
    }
public:

    std::vector< sequence<2,size_t> > keys;
    std::vector< sparse_bispace<1> > subspaces;
    std::vector< sparse_block_tree::value_t > values;
    two_d_test_f() : keys(init_keys()),subspaces(init_subspaces()),values(init_values()) {}
};

class three_d_test_f {
    static std::vector< sequence<3,size_t> > init_keys() 
    {
        //All 3d blocks are size 8
        size_t seq01_arr[3] = {1,2,3}; //offset 0
        size_t seq02_arr[3] = {1,2,7}; //offset 8
        size_t seq03_arr[3] = {1,3,1}; //offset 16
        size_t seq04_arr[3] = {1,5,9}; //offset 24
        size_t seq05_arr[3] = {2,3,1}; //offset 32
        size_t seq06_arr[3] = {2,4,2}; //offset 40
        size_t seq07_arr[3] = {2,4,5}; //offset 48
        size_t seq08_arr[3] = {2,6,3}; //offset 56
        size_t seq09_arr[3] = {2,6,4}; //offset 64
        size_t seq10_arr[3] = {4,1,4}; //offset 72
        size_t seq11_arr[3] = {4,1,7}; //offset 80
        size_t seq12_arr[3] = {4,2,2}; //offset 88
        size_t seq13_arr[3] = {4,3,5}; //offset 96
        size_t seq14_arr[3] = {4,3,6}; //offset 104
        size_t seq15_arr[3] = {4,3,7}; //offset 112
        size_t seq16_arr[3] = {5,1,4}; //offset 120
        size_t seq17_arr[3] = {5,2,6}; //offset 128
        size_t seq18_arr[3] = {5,2,7}; //offset 136
        size_t seq19_arr[3] = {7,4,5}; //offset 144
        size_t seq20_arr[3] = {7,4,6}; //offset 152
        size_t seq21_arr[3] = {7,7,7}; //offset 160

        std::vector< sequence<3,size_t> > key_list(21); 
        for(size_t i = 0; i < 3; ++i) key_list[0][i] = seq01_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[1][i] = seq02_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[2][i] = seq03_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[3][i] = seq04_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[4][i] = seq05_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[5][i] = seq06_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[6][i] = seq07_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[7][i] = seq08_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[8][i] = seq09_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[9][i] = seq10_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[10][i] = seq11_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[11][i] = seq12_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[12][i] = seq13_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[13][i] = seq14_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[14][i] = seq15_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[15][i] = seq16_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[16][i] = seq17_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[17][i] = seq18_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[18][i] = seq19_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[19][i] = seq20_arr[i];
        for(size_t i = 0; i < 3; ++i) key_list[20][i] = seq21_arr[i];

        return key_list;
    }

    static std::vector< sparse_bispace<1> > init_subspaces()
    {
        sparse_bispace<1> spb_1(20);
        vector<size_t> split_points_1;
        for(size_t i = 2; i < 20; i += 2)
        {
            split_points_1.push_back(i);
        }
        spb_1.split(split_points_1);

        return std::vector< sparse_bispace<1> >(3,spb_1);
    }

    static std::vector< sparse_block_tree::value_t > init_values()
    {
        size_t offset_arr[21] = {0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160};
        size_t size_arr[21] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
        vector< sparse_block_tree::value_t > values;
        for(size_t i = 0; i < 21; ++i)
        {
            values.push_back(sparse_block_tree::value_t(1,off_dim_pair(offset_arr[i],size_arr[i])));
        }
        return values;
    }
public:

    std::vector< sequence<3,size_t> > keys;
    std::vector< sparse_bispace<1> > subspaces;
    std::vector< sparse_block_tree::value_t > values;
    three_d_test_f() : keys(init_keys()),subspaces(init_subspaces()),values(init_values()) {}
};

//Used to check the keys and values of a tree
//Cleaner for test implementation than using an equality operator because
//everything can just be set up as vectors instead of constructing a benchmark tree with sequences
bool verify_tree(const sparse_block_tree& tree,const std::vector< std::vector<size_t> >& correct_keys, const std::vector< sparse_block_tree::value_t >& correct_values)
{
    size_t m = 0;
    for(sparse_block_tree::const_iterator it = tree.begin(); it != tree.end(); ++it)
    {
        if(it.key() != correct_keys[m])
        {
            return false;
        }
        if(*it != correct_values[m])
        {
            return false;
        }
        ++m;
    }
    if(m != correct_keys.size())
    {
        return false;
    }
    return true;
}

} // namespace unnamed

void sparse_block_tree_test::perform() throw(libtest::test_exception)
{
    test_zero_order();
    test_unsorted_input();

    test_get_nnz_2d();
    test_get_n_entries_3d();

    test_equality_false_2d();
    test_equality_true_2d();
    
    test_search_2d_invalid_key();
    test_search_3d();

    test_permute_3d();

    test_get_sub_tree_invalid_key_size();
    test_get_sub_tree_nonexistent_key();
    test_get_sub_tree_2d();
    test_get_sub_tree_3d();

    test_contract_3d_0();
    test_contract_3d_1();
    test_contract_3d_2();

    test_fuse_3d_2d();
    test_fuse_3d_3d_non_contiguous();
    test_fuse_3d_3d_multi_index();

    test_truncate_subspace_3d();

    test_insert_subspace_3d();
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
        sparse_block_tree sbt(block_tuples_list,std::vector< sparse_bispace<1> >());
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

void sparse_block_tree_test::test_unsorted_input() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_unsorted_input()";

    two_d_test_f tf = two_d_test_f();
    //Corrupt one entry to make the list unsorted
    tf.keys[3][1] = 5;

    bool threw_exception = false;
    try
    {
        sparse_block_tree sbt(tf.keys,tf.subspaces);
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

void sparse_block_tree_test::test_get_nnz_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_nnz_2d()";

    two_d_test_f tf = two_d_test_f();
    
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);
    
    if(sbt_1.get_nnz() != 32)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree::get_nnz(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_get_n_entries_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_n_entries_3d()";

    three_d_test_f tf = three_d_test_f();
    
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);
    
    if(sbt_1.get_n_entries() != 21)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree::get_n_entries(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_equality_false_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_equality_false_2d()";

    two_d_test_f tf = two_d_test_f();
    
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);
    
    //Change one entry
    tf.keys[2][1] += 1;
    sparse_block_tree sbt_2(tf.keys,tf.subspaces);
    
    if(sbt_1 == sbt_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree::operator==(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_equality_true_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_equality_true_2d()";

    two_d_test_f tf = two_d_test_f();
    
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);
    sparse_block_tree sbt_2(tf.keys,tf.subspaces);
    
    if(sbt_1 != sbt_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparse_block_tree::operator==(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_search_2d_invalid_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_search_2d_invalid_key()";

    two_d_test_f tf = two_d_test_f();
    const sparse_block_tree sbt(tf.keys,tf.subspaces);

    std::vector<size_t> key(2);
    key[0] = 1; 
    key[1] = 3; // (1,3) is not in tree
    sparse_block_tree::const_iterator it = sbt.search(key);

    if(it != sbt.end())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree::search(...) did not return end() for invalid key");
    }
}

void sparse_block_tree_test::test_search_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_search_3d()";

    three_d_test_f tf = three_d_test_f();

    sparse_block_tree sbt(tf.keys,tf.subspaces);

    std::vector<size_t> key(3);
    key[0] = 4; 
    key[1] = 1;
    key[2] = 4;
    
    sparse_block_tree::value_t correct_val = sparse_block_tree::value_t(1,make_pair(72,8));
    if(*sbt.search(key) != correct_val)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree::search(...) did not return correct value");
    }
}

void sparse_block_tree_test::test_permute_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_permute_3d()";

    three_d_test_f tf = three_d_test_f();

    const sparse_block_tree sbt(tf.keys,tf.subspaces);

    //permute first and last index
    runtime_permutation perm(3);
    perm.permute(0,2);

    const sparse_block_tree permuted_sbt = sbt.permute(perm);

    //Benchmark result
    std::vector< block_list > correct_keys;
    size_t correct_seq00_arr[3] = {1,3,1}; correct_keys.push_back(block_list(&correct_seq00_arr[0],&correct_seq00_arr[0]+3));
    size_t correct_seq01_arr[3] = {1,3,2}; correct_keys.push_back(block_list(&correct_seq01_arr[0],&correct_seq01_arr[0]+3));
    size_t correct_seq02_arr[3] = {2,2,4}; correct_keys.push_back(block_list(&correct_seq02_arr[0],&correct_seq02_arr[0]+3));
    size_t correct_seq03_arr[3] = {2,4,2}; correct_keys.push_back(block_list(&correct_seq03_arr[0],&correct_seq03_arr[0]+3));
    size_t correct_seq04_arr[3] = {3,2,1}; correct_keys.push_back(block_list(&correct_seq04_arr[0],&correct_seq04_arr[0]+3));
    size_t correct_seq05_arr[3] = {3,6,2}; correct_keys.push_back(block_list(&correct_seq05_arr[0],&correct_seq05_arr[0]+3));
    size_t correct_seq06_arr[3] = {4,1,4}; correct_keys.push_back(block_list(&correct_seq06_arr[0],&correct_seq06_arr[0]+3));
    size_t correct_seq07_arr[3] = {4,1,5}; correct_keys.push_back(block_list(&correct_seq07_arr[0],&correct_seq07_arr[0]+3));
    size_t correct_seq08_arr[3] = {4,6,2}; correct_keys.push_back(block_list(&correct_seq08_arr[0],&correct_seq08_arr[0]+3));
    size_t correct_seq09_arr[3] = {5,3,4}; correct_keys.push_back(block_list(&correct_seq09_arr[0],&correct_seq09_arr[0]+3));
    size_t correct_seq10_arr[3] = {5,4,2}; correct_keys.push_back(block_list(&correct_seq10_arr[0],&correct_seq10_arr[0]+3));
    size_t correct_seq11_arr[3] = {5,4,7}; correct_keys.push_back(block_list(&correct_seq11_arr[0],&correct_seq11_arr[0]+3));
    size_t correct_seq12_arr[3] = {6,2,5}; correct_keys.push_back(block_list(&correct_seq12_arr[0],&correct_seq12_arr[0]+3));
    size_t correct_seq13_arr[3] = {6,3,4}; correct_keys.push_back(block_list(&correct_seq13_arr[0],&correct_seq13_arr[0]+3));
    size_t correct_seq14_arr[3] = {6,4,7}; correct_keys.push_back(block_list(&correct_seq14_arr[0],&correct_seq14_arr[0]+3));
    size_t correct_seq15_arr[3] = {7,1,4}; correct_keys.push_back(block_list(&correct_seq15_arr[0],&correct_seq15_arr[0]+3));
    size_t correct_seq16_arr[3] = {7,2,1}; correct_keys.push_back(block_list(&correct_seq16_arr[0],&correct_seq16_arr[0]+3));
    size_t correct_seq17_arr[3] = {7,2,5}; correct_keys.push_back(block_list(&correct_seq17_arr[0],&correct_seq17_arr[0]+3));
    size_t correct_seq18_arr[3] = {7,3,4}; correct_keys.push_back(block_list(&correct_seq18_arr[0],&correct_seq18_arr[0]+3));
    size_t correct_seq19_arr[3] = {7,7,7}; correct_keys.push_back(block_list(&correct_seq19_arr[0],&correct_seq19_arr[0]+3));
    size_t correct_seq20_arr[3] = {9,5,1}; correct_keys.push_back(block_list(&correct_seq20_arr[0],&correct_seq20_arr[0]+3));

    size_t correct_offset_arr[21] = {16,32,88,40,0,56,72,120,64,96,48,144,128,104,152,80,8,136,112,160,24};
    size_t correct_size_arr[21] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
    std::vector< sparse_block_tree::value_t > correct_vals; 
    for(size_t i = 0; i < 21; ++i) correct_vals.push_back(sparse_block_tree::value_t(1,off_dim_pair(correct_offset_arr[i],correct_size_arr[i])));

    if(!verify_tree(permuted_sbt,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "permute returned invalid tree");
    }

}

void sparse_block_tree_test::test_get_sub_tree_invalid_key_size() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_tree_invalid_key_size()";

    two_d_test_f tf = two_d_test_f();

    sparse_block_tree sbt(tf.keys,tf.subspaces);

    //First test key that is too long
    bool threw_exception = false;
    try
    {
        std::vector<size_t> too_big_key;
        too_big_key.push_back(5);
        too_big_key.push_back(2);
        sbt.get_sub_tree(too_big_key);
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
        sbt.get_sub_tree(std::vector<size_t>());
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

void sparse_block_tree_test::test_get_sub_tree_nonexistent_key() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_tree_nonexistent_key()";

    two_d_test_f tf = two_d_test_f();
    sparse_block_tree sbt(tf.keys,tf.subspaces);

    std::vector<size_t> nonexistent_key(1,6);
    const sparse_block_tree& sub_tree = sbt.get_sub_tree(nonexistent_key);

    if(sub_tree.begin() != sub_tree.end())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "did not return empty tree for key that does not exist!");
    }
}

void sparse_block_tree_test::test_get_sub_tree_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_tree_2d()";

    two_d_test_f tf = two_d_test_f();
    sparse_block_tree sbt(tf.keys,tf.subspaces);

    const sparse_block_tree& st = sbt.get_sub_tree(std::vector<size_t>(1,4));

    std::vector< std::vector<size_t> > correct_keys(1,std::vector<size_t>(1,1)); 
    correct_keys.push_back(std::vector<size_t>(1,4));
    std::vector< sparse_block_tree::value_t > correct_vals(tf.values.begin()+3,tf.values.begin()+5);
    size_t m = 0;
    for(sparse_block_tree::const_iterator it = st.begin(); it != st.end(); ++it)
    {
        if(it.key() != correct_keys[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sub tree iterator returned wrong key");
        }
        if(*it != correct_vals[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "*it for sub tree corresponding to 1d key is incorrect");
        }
        ++m;
    }
    if(m != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sub tree did not contain enough keys");
    }
} 

void sparse_block_tree_test::test_get_sub_tree_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_get_sub_tree_3d()";

    three_d_test_f tf = three_d_test_f();
    sparse_block_tree sbt(tf.keys,tf.subspaces);

    //Test a 1D key
    const sparse_block_tree& sub_tree_1 = sbt.get_sub_tree(std::vector<size_t>(1,2));

    //Correct result for 1d key
    std::vector< block_list > correct_keys_1;
    size_t correct_seq0_arr_1[2] = {3,1}; correct_keys_1.push_back(block_list(&correct_seq0_arr_1[0],&correct_seq0_arr_1[0]+2));
    size_t correct_seq1_arr_1[2] = {4,2}; correct_keys_1.push_back(block_list(&correct_seq1_arr_1[0],&correct_seq1_arr_1[0]+2));
    size_t correct_seq2_arr_1[2] = {4,5}; correct_keys_1.push_back(block_list(&correct_seq2_arr_1[0],&correct_seq2_arr_1[0]+2));
    size_t correct_seq3_arr_1[2] = {6,3}; correct_keys_1.push_back(block_list(&correct_seq3_arr_1[0],&correct_seq3_arr_1[0]+2));
    size_t correct_seq4_arr_1[2] = {6,4}; correct_keys_1.push_back(block_list(&correct_seq4_arr_1[0],&correct_seq4_arr_1[0]+2));

    std::vector< sparse_block_tree::value_t > correct_vals_1(tf.values.begin()+4,tf.values.begin()+9);

    size_t m = 0;
    for(sparse_block_tree::const_iterator it = sub_tree_1.begin(); it != sub_tree_1.end(); ++it)
    {
        if(it.key() != correct_keys_1[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "it.key() for sub tree corresponding to 1d key is incorrect");
        }
        if(*it != correct_vals_1[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "*it for sub tree corresponding to 1d key is incorrect");
        }
        ++m;
    }
    if(m != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "++it did not traverse enough keys");
    }

    //Test a 2d key
    std::vector<size_t> key_2d(1,2);
    key_2d.push_back(6);
    const sparse_block_tree& sub_tree_2 = sbt.get_sub_tree(key_2d);

    //Correct result for 2d key
    std::vector< std::vector<size_t> > correct_keys_2(2);
    correct_keys_2[0].push_back(3);
    correct_keys_2[1].push_back(4);
    std::vector< sparse_block_tree::value_t > correct_vals_2(tf.values.begin()+7,tf.values.begin()+9);

    m = 0;
    for(sparse_block_tree::const_iterator it = sub_tree_2.begin(); it != sub_tree_2.end(); ++it)
    {
        if(it.key() != correct_keys_2[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "it.key() for sub tree corresponding to 1d key is incorrect");
        }
        if(*it != correct_vals_2[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "*it() for sub tree corresponding to 1d key is incorrect");
        }
        ++m;
    }
    if(m != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "++it did not traverse enough keys");
    }
}

void sparse_block_tree_test::test_contract_3d_0() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_0()";

    three_d_test_f tf = three_d_test_f();

    sparse_block_tree sbt(tf.keys,tf.subspaces);
    std::vector< sparse_bispace<1> > contracted_subspaces(1,tf.subspaces[1]);
    contracted_subspaces.push_back(tf.subspaces[2]);
    sparse_block_tree sbt_contracted = sbt.contract(0,contracted_subspaces);

    //Build the correct tree
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_seq00_arr[2] = {1,4}; correct_keys.push_back(block_list(&correct_seq00_arr[0],&correct_seq00_arr[0]+2));
    size_t correct_seq01_arr[2] = {1,7}; correct_keys.push_back(block_list(&correct_seq01_arr[0],&correct_seq01_arr[0]+2));
    size_t correct_seq02_arr[2] = {2,2}; correct_keys.push_back(block_list(&correct_seq02_arr[0],&correct_seq02_arr[0]+2));
    size_t correct_seq03_arr[2] = {2,3}; correct_keys.push_back(block_list(&correct_seq03_arr[0],&correct_seq03_arr[0]+2));
    size_t correct_seq04_arr[2] = {2,6}; correct_keys.push_back(block_list(&correct_seq04_arr[0],&correct_seq04_arr[0]+2));
    size_t correct_seq05_arr[2] = {2,7}; correct_keys.push_back(block_list(&correct_seq05_arr[0],&correct_seq05_arr[0]+2));
    size_t correct_seq06_arr[2] = {3,1}; correct_keys.push_back(block_list(&correct_seq06_arr[0],&correct_seq06_arr[0]+2));
    size_t correct_seq07_arr[2] = {3,5}; correct_keys.push_back(block_list(&correct_seq07_arr[0],&correct_seq07_arr[0]+2));
    size_t correct_seq08_arr[2] = {3,6}; correct_keys.push_back(block_list(&correct_seq08_arr[0],&correct_seq08_arr[0]+2));
    size_t correct_seq09_arr[2] = {3,7}; correct_keys.push_back(block_list(&correct_seq09_arr[0],&correct_seq09_arr[0]+2));
    size_t correct_seq10_arr[2] = {4,2}; correct_keys.push_back(block_list(&correct_seq10_arr[0],&correct_seq10_arr[0]+2));
    size_t correct_seq11_arr[2] = {4,5}; correct_keys.push_back(block_list(&correct_seq11_arr[0],&correct_seq11_arr[0]+2));
    size_t correct_seq12_arr[2] = {4,6}; correct_keys.push_back(block_list(&correct_seq12_arr[0],&correct_seq12_arr[0]+2));
    size_t correct_seq13_arr[2] = {5,9}; correct_keys.push_back(block_list(&correct_seq13_arr[0],&correct_seq13_arr[0]+2));
    size_t correct_seq14_arr[2] = {6,3}; correct_keys.push_back(block_list(&correct_seq14_arr[0],&correct_seq14_arr[0]+2));
    size_t correct_seq15_arr[2] = {6,4}; correct_keys.push_back(block_list(&correct_seq15_arr[0],&correct_seq15_arr[0]+2));
    size_t correct_seq16_arr[2] = {7,7}; correct_keys.push_back(block_list(&correct_seq16_arr[0],&correct_seq16_arr[0]+2));

    
    size_t correct_offset_arr[17] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64};
    size_t correct_size_arr[17] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};
    std::vector< sparse_block_tree::value_t > correct_vals; 
    for(size_t i = 0; i < 17; ++i) correct_vals.push_back(sparse_block_tree::value_t(1,off_dim_pair(correct_offset_arr[i],correct_size_arr[i])));

    if(!verify_tree(sbt_contracted,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::contract(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_contract_3d_1() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_1()";

    three_d_test_f tf = three_d_test_f();

    sparse_block_tree sbt(tf.keys,tf.subspaces);
    std::vector< sparse_bispace<1> > contracted_subspaces(1,tf.subspaces[0]);
    contracted_subspaces.push_back(tf.subspaces[2]);
    sparse_block_tree sbt_contracted = sbt.contract(1,contracted_subspaces);

    //Build the correct tree
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_seq00_arr[2] = {1,1}; correct_keys.push_back(block_list(&correct_seq00_arr[0],&correct_seq00_arr[0]+2));
    size_t correct_seq01_arr[2] = {1,3}; correct_keys.push_back(block_list(&correct_seq01_arr[0],&correct_seq01_arr[0]+2));
    size_t correct_seq02_arr[2] = {1,7}; correct_keys.push_back(block_list(&correct_seq02_arr[0],&correct_seq02_arr[0]+2));
    size_t correct_seq03_arr[2] = {1,9}; correct_keys.push_back(block_list(&correct_seq03_arr[0],&correct_seq03_arr[0]+2));
    size_t correct_seq04_arr[2] = {2,1}; correct_keys.push_back(block_list(&correct_seq04_arr[0],&correct_seq04_arr[0]+2));
    size_t correct_seq05_arr[2] = {2,2}; correct_keys.push_back(block_list(&correct_seq05_arr[0],&correct_seq05_arr[0]+2));
    size_t correct_seq06_arr[2] = {2,3}; correct_keys.push_back(block_list(&correct_seq06_arr[0],&correct_seq06_arr[0]+2));
    size_t correct_seq07_arr[2] = {2,4}; correct_keys.push_back(block_list(&correct_seq07_arr[0],&correct_seq07_arr[0]+2));
    size_t correct_seq08_arr[2] = {2,5}; correct_keys.push_back(block_list(&correct_seq08_arr[0],&correct_seq08_arr[0]+2));
    size_t correct_seq09_arr[2] = {4,2}; correct_keys.push_back(block_list(&correct_seq09_arr[0],&correct_seq09_arr[0]+2));
    size_t correct_seq10_arr[2] = {4,4}; correct_keys.push_back(block_list(&correct_seq10_arr[0],&correct_seq10_arr[0]+2));
    size_t correct_seq11_arr[2] = {4,5}; correct_keys.push_back(block_list(&correct_seq11_arr[0],&correct_seq11_arr[0]+2));
    size_t correct_seq12_arr[2] = {4,6}; correct_keys.push_back(block_list(&correct_seq12_arr[0],&correct_seq12_arr[0]+2));
    size_t correct_seq13_arr[2] = {4,7}; correct_keys.push_back(block_list(&correct_seq13_arr[0],&correct_seq13_arr[0]+2));
    size_t correct_seq14_arr[2] = {5,4}; correct_keys.push_back(block_list(&correct_seq14_arr[0],&correct_seq14_arr[0]+2));
    size_t correct_seq15_arr[2] = {5,6}; correct_keys.push_back(block_list(&correct_seq15_arr[0],&correct_seq15_arr[0]+2));
    size_t correct_seq16_arr[2] = {5,7}; correct_keys.push_back(block_list(&correct_seq16_arr[0],&correct_seq16_arr[0]+2));
    size_t correct_seq17_arr[2] = {7,5}; correct_keys.push_back(block_list(&correct_seq17_arr[0],&correct_seq17_arr[0]+2));
    size_t correct_seq18_arr[2] = {7,6}; correct_keys.push_back(block_list(&correct_seq18_arr[0],&correct_seq18_arr[0]+2));
    size_t correct_seq19_arr[2] = {7,7}; correct_keys.push_back(block_list(&correct_seq19_arr[0],&correct_seq19_arr[0]+2));

    size_t correct_offset_arr[20] = {0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76};
    size_t correct_size_arr[20] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};
    std::vector< sparse_block_tree::value_t > correct_vals; 
    for(size_t i = 0; i < 20; ++i) correct_vals.push_back(sparse_block_tree::value_t(1,off_dim_pair(correct_offset_arr[i],correct_size_arr[i])));

    if(!verify_tree(sbt_contracted,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::contract(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_contract_3d_2() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_contract_3d_2()";

    three_d_test_f tf = three_d_test_f();

    sparse_block_tree sbt(tf.keys,tf.subspaces);
    std::vector< sparse_bispace<1> > contracted_subspaces(1,tf.subspaces[0]);
    contracted_subspaces.push_back(tf.subspaces[1]);
    sparse_block_tree sbt_contracted = sbt.contract(2,contracted_subspaces);

    //Build the correct tree
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_seq00_arr[2] = {1,2}; correct_keys.push_back(block_list(&correct_seq00_arr[0],&correct_seq00_arr[0]+2));
    size_t correct_seq01_arr[2] = {1,3}; correct_keys.push_back(block_list(&correct_seq01_arr[0],&correct_seq01_arr[0]+2));
    size_t correct_seq02_arr[2] = {1,5}; correct_keys.push_back(block_list(&correct_seq02_arr[0],&correct_seq02_arr[0]+2));
    size_t correct_seq03_arr[2] = {2,3}; correct_keys.push_back(block_list(&correct_seq03_arr[0],&correct_seq03_arr[0]+2));
    size_t correct_seq04_arr[2] = {2,4}; correct_keys.push_back(block_list(&correct_seq04_arr[0],&correct_seq04_arr[0]+2));
    size_t correct_seq05_arr[2] = {2,6}; correct_keys.push_back(block_list(&correct_seq05_arr[0],&correct_seq05_arr[0]+2));
    size_t correct_seq06_arr[2] = {4,1}; correct_keys.push_back(block_list(&correct_seq06_arr[0],&correct_seq06_arr[0]+2));
    size_t correct_seq07_arr[2] = {4,2}; correct_keys.push_back(block_list(&correct_seq07_arr[0],&correct_seq07_arr[0]+2));
    size_t correct_seq08_arr[2] = {4,3}; correct_keys.push_back(block_list(&correct_seq08_arr[0],&correct_seq08_arr[0]+2));
    size_t correct_seq09_arr[2] = {5,1}; correct_keys.push_back(block_list(&correct_seq09_arr[0],&correct_seq09_arr[0]+2));
    size_t correct_seq10_arr[2] = {5,2}; correct_keys.push_back(block_list(&correct_seq10_arr[0],&correct_seq10_arr[0]+2));
    size_t correct_seq11_arr[2] = {7,4}; correct_keys.push_back(block_list(&correct_seq11_arr[0],&correct_seq11_arr[0]+2));
    size_t correct_seq12_arr[2] = {7,7}; correct_keys.push_back(block_list(&correct_seq12_arr[0],&correct_seq12_arr[0]+2));

    size_t correct_offset_arr[13] = {0,4,8,12,16,20,24,28,32,36,40,44,48};
    size_t correct_size_arr[13] = {4,4,4,4,4,4,4,4,4,4,4,4,4};
    std::vector< sparse_block_tree::value_t > correct_vals; 
    for(size_t i = 0; i < 13; ++i) correct_vals.push_back(sparse_block_tree::value_t(1,off_dim_pair(correct_offset_arr[i],correct_size_arr[i])));

    if(!verify_tree(sbt_contracted,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::contract(...) returned incorrect value");
    }
}

//This test is important because it shows how the sizes of the blocks in the different
//trees are correctly saved by fuse
void sparse_block_tree_test::test_fuse_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_2d()";

    three_d_test_f tf_3d =  three_d_test_f();
    sparse_block_tree sbt_1(tf_3d.keys,tf_3d.subspaces);

    two_d_test_f tf_2d = two_d_test_f();
    sparse_block_tree sbt_2(tf_2d.keys,tf_2d.subspaces);

    sparse_block_tree sbt_fused = sbt_1.fuse(sbt_2); 

    //Correct keys
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_key00_arr[4] = {1,2,7,4}; correct_keys.push_back(block_list(&correct_key00_arr[0],&correct_key00_arr[0]+4));
    size_t correct_key01_arr[4] = {1,3,1,2}; correct_keys.push_back(block_list(&correct_key01_arr[0],&correct_key01_arr[0]+4));
    size_t correct_key02_arr[4] = {1,3,1,5}; correct_keys.push_back(block_list(&correct_key02_arr[0],&correct_key02_arr[0]+4));
    size_t correct_key03_arr[4] = {2,3,1,2}; correct_keys.push_back(block_list(&correct_key03_arr[0],&correct_key03_arr[0]+4));
    size_t correct_key04_arr[4] = {2,3,1,5}; correct_keys.push_back(block_list(&correct_key04_arr[0],&correct_key04_arr[0]+4));
    size_t correct_key05_arr[4] = {2,4,2,3}; correct_keys.push_back(block_list(&correct_key05_arr[0],&correct_key05_arr[0]+4));
    size_t correct_key06_arr[4] = {2,4,5,1}; correct_keys.push_back(block_list(&correct_key06_arr[0],&correct_key06_arr[0]+4));
    size_t correct_key07_arr[4] = {2,4,5,2}; correct_keys.push_back(block_list(&correct_key07_arr[0],&correct_key07_arr[0]+4));
    size_t correct_key08_arr[4] = {2,6,4,1}; correct_keys.push_back(block_list(&correct_key08_arr[0],&correct_key08_arr[0]+4));
    size_t correct_key09_arr[4] = {2,6,4,4}; correct_keys.push_back(block_list(&correct_key09_arr[0],&correct_key09_arr[0]+4));
    size_t correct_key10_arr[4] = {4,1,4,1}; correct_keys.push_back(block_list(&correct_key10_arr[0],&correct_key10_arr[0]+4));
    size_t correct_key11_arr[4] = {4,1,4,4}; correct_keys.push_back(block_list(&correct_key11_arr[0],&correct_key11_arr[0]+4));
    size_t correct_key12_arr[4] = {4,1,7,4}; correct_keys.push_back(block_list(&correct_key12_arr[0],&correct_key12_arr[0]+4));
    size_t correct_key13_arr[4] = {4,2,2,3}; correct_keys.push_back(block_list(&correct_key13_arr[0],&correct_key13_arr[0]+4));
    size_t correct_key14_arr[4] = {4,3,5,1}; correct_keys.push_back(block_list(&correct_key14_arr[0],&correct_key14_arr[0]+4));
    size_t correct_key15_arr[4] = {4,3,5,2}; correct_keys.push_back(block_list(&correct_key15_arr[0],&correct_key15_arr[0]+4));
    size_t correct_key16_arr[4] = {4,3,7,4}; correct_keys.push_back(block_list(&correct_key16_arr[0],&correct_key16_arr[0]+4));
    size_t correct_key17_arr[4] = {5,1,4,1}; correct_keys.push_back(block_list(&correct_key17_arr[0],&correct_key17_arr[0]+4));
    size_t correct_key18_arr[4] = {5,1,4,4}; correct_keys.push_back(block_list(&correct_key18_arr[0],&correct_key18_arr[0]+4));
    size_t correct_key19_arr[4] = {5,2,7,4}; correct_keys.push_back(block_list(&correct_key19_arr[0],&correct_key19_arr[0]+4));
    size_t correct_key20_arr[4] = {7,4,5,1}; correct_keys.push_back(block_list(&correct_key20_arr[0],&correct_key20_arr[0]+4));
    size_t correct_key21_arr[4] = {7,4,5,2}; correct_keys.push_back(block_list(&correct_key21_arr[0],&correct_key21_arr[0]+4));
    size_t correct_key22_arr[4] = {7,7,7,4}; correct_keys.push_back(block_list(&correct_key22_arr[0],&correct_key22_arr[0]+4));

    //Correct values
    std::vector< sparse_block_tree::value_t > correct_vals;
    off_dim_pair correct_value00_arr[2] = { off_dim_pair(8,8),off_dim_pair(28,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value00_arr[0],&correct_value00_arr[0]+2));
    off_dim_pair correct_value01_arr[2] = { off_dim_pair(16,8),off_dim_pair(0,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value01_arr[0],&correct_value01_arr[0]+2));
    off_dim_pair correct_value02_arr[2] = { off_dim_pair(16,8),off_dim_pair(4,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value02_arr[0],&correct_value02_arr[0]+2));
    off_dim_pair correct_value03_arr[2] = { off_dim_pair(32,8),off_dim_pair(0,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value03_arr[0],&correct_value03_arr[0]+2));
    off_dim_pair correct_value04_arr[2] = { off_dim_pair(32,8),off_dim_pair(4,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value04_arr[0],&correct_value04_arr[0]+2));
    off_dim_pair correct_value05_arr[2] = { off_dim_pair(40,8),off_dim_pair(8,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value05_arr[0],&correct_value05_arr[0]+2));
    off_dim_pair correct_value06_arr[2] = { off_dim_pair(48,8),off_dim_pair(20,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value06_arr[0],&correct_value06_arr[0]+2));
    off_dim_pair correct_value07_arr[2] = { off_dim_pair(48,8),off_dim_pair(24,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value07_arr[0],&correct_value07_arr[0]+2));
    off_dim_pair correct_value08_arr[2] = { off_dim_pair(64,8),off_dim_pair(12,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value08_arr[0],&correct_value08_arr[0]+2));
    off_dim_pair correct_value09_arr[2] = { off_dim_pair(64,8),off_dim_pair(16,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value09_arr[0],&correct_value09_arr[0]+2));
    off_dim_pair correct_value10_arr[2] = { off_dim_pair(72,8),off_dim_pair(12,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value10_arr[0],&correct_value10_arr[0]+2));
    off_dim_pair correct_value11_arr[2] = { off_dim_pair(72,8),off_dim_pair(16,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value11_arr[0],&correct_value11_arr[0]+2));
    off_dim_pair correct_value12_arr[2] = { off_dim_pair(80,8),off_dim_pair(28,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value12_arr[0],&correct_value12_arr[0]+2));
    off_dim_pair correct_value13_arr[2] = { off_dim_pair(88,8),off_dim_pair(8,4) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value13_arr[0],&correct_value13_arr[0]+2));
    off_dim_pair correct_value14_arr[2] = { off_dim_pair(96,8),off_dim_pair(20,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value14_arr[0],&correct_value14_arr[0]+2));
    off_dim_pair correct_value15_arr[2] = { off_dim_pair(96,8),off_dim_pair(24,4) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value15_arr[0],&correct_value15_arr[0]+2));
    off_dim_pair correct_value16_arr[2] = { off_dim_pair(112,8),off_dim_pair(28,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value16_arr[0],&correct_value16_arr[0]+2));
    off_dim_pair correct_value17_arr[2] = { off_dim_pair(120,8),off_dim_pair(12,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value17_arr[0],&correct_value17_arr[0]+2));
    off_dim_pair correct_value18_arr[2] = { off_dim_pair(120,8),off_dim_pair(16,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value18_arr[0],&correct_value18_arr[0]+2));
    off_dim_pair correct_value19_arr[2] = { off_dim_pair(136,8),off_dim_pair(28,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value19_arr[0],&correct_value19_arr[0]+2));
    off_dim_pair correct_value20_arr[2] = { off_dim_pair(144,8),off_dim_pair(20,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value20_arr[0],&correct_value20_arr[0]+2));
    off_dim_pair correct_value21_arr[2] = { off_dim_pair(144,8),off_dim_pair(24,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value21_arr[0],&correct_value21_arr[0]+2));
    off_dim_pair correct_value22_arr[2] = { off_dim_pair(160,8),off_dim_pair(28,4) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value22_arr[0],&correct_value22_arr[0]+2));

    if(!verify_tree(sbt_fused,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::fuse(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_fuse_3d_3d_non_contiguous() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_3d_non_contiguous()";

    //Work with a reduced subset of the keys to keep the result manageable
    three_d_test_f tf = three_d_test_f();
    std::vector< sequence<3,size_t> > keys(tf.keys.begin(),tf.keys.begin()+11);
    sparse_block_tree sbt_1(keys,tf.subspaces);
    sparse_block_tree sbt_2(keys,tf.subspaces);

    sparse_block_tree sbt_fused = sbt_1.fuse(sbt_2,idx_list(1,2),idx_list(1,1)); 

    //Correct keys
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_key00_arr[5] = {1,2,3,1,1}; correct_keys.push_back(block_list(&correct_key00_arr[0],&correct_key00_arr[0]+5));
    size_t correct_key01_arr[5] = {1,2,3,2,1}; correct_keys.push_back(block_list(&correct_key01_arr[0],&correct_key01_arr[0]+5));
    size_t correct_key02_arr[5] = {1,3,1,4,4}; correct_keys.push_back(block_list(&correct_key02_arr[0],&correct_key02_arr[0]+5));
    size_t correct_key03_arr[5] = {1,3,1,4,7}; correct_keys.push_back(block_list(&correct_key03_arr[0],&correct_key03_arr[0]+5));
    size_t correct_key04_arr[5] = {2,3,1,4,4}; correct_keys.push_back(block_list(&correct_key04_arr[0],&correct_key04_arr[0]+5));
    size_t correct_key05_arr[5] = {2,3,1,4,7}; correct_keys.push_back(block_list(&correct_key05_arr[0],&correct_key05_arr[0]+5));
    size_t correct_key06_arr[5] = {2,4,2,1,3}; correct_keys.push_back(block_list(&correct_key06_arr[0],&correct_key06_arr[0]+5));
    size_t correct_key07_arr[5] = {2,4,2,1,7}; correct_keys.push_back(block_list(&correct_key07_arr[0],&correct_key07_arr[0]+5));
    size_t correct_key08_arr[5] = {2,4,5,1,9}; correct_keys.push_back(block_list(&correct_key08_arr[0],&correct_key08_arr[0]+5));
    size_t correct_key09_arr[5] = {2,6,3,1,1}; correct_keys.push_back(block_list(&correct_key09_arr[0],&correct_key09_arr[0]+5));
    size_t correct_key10_arr[5] = {2,6,3,2,1}; correct_keys.push_back(block_list(&correct_key10_arr[0],&correct_key10_arr[0]+5));
    size_t correct_key11_arr[5] = {2,6,4,2,2}; correct_keys.push_back(block_list(&correct_key11_arr[0],&correct_key11_arr[0]+5));
    size_t correct_key12_arr[5] = {2,6,4,2,5}; correct_keys.push_back(block_list(&correct_key12_arr[0],&correct_key12_arr[0]+5));
    size_t correct_key13_arr[5] = {4,1,4,2,2}; correct_keys.push_back(block_list(&correct_key13_arr[0],&correct_key13_arr[0]+5));
    size_t correct_key14_arr[5] = {4,1,4,2,5}; correct_keys.push_back(block_list(&correct_key14_arr[0],&correct_key14_arr[0]+5));

    //Correct values
    std::vector< sparse_block_tree::value_t > correct_vals;
    off_dim_pair correct_value00_arr[2] = { off_dim_pair(0,8),off_dim_pair(16,8) };    correct_vals.push_back(sparse_block_tree::value_t(&correct_value00_arr[0],&correct_value00_arr[0]+2));
    off_dim_pair correct_value01_arr[2] = { off_dim_pair(0,8),off_dim_pair(32,8) };    correct_vals.push_back(sparse_block_tree::value_t(&correct_value01_arr[0],&correct_value01_arr[0]+2));
    off_dim_pair correct_value02_arr[2] = { off_dim_pair(16,8),off_dim_pair(72,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value02_arr[0],&correct_value02_arr[0]+2));
    off_dim_pair correct_value03_arr[2] = { off_dim_pair(16,8),off_dim_pair(80,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value03_arr[0],&correct_value03_arr[0]+2));
    off_dim_pair correct_value04_arr[2] = { off_dim_pair(32,8),off_dim_pair(72,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value04_arr[0],&correct_value04_arr[0]+2));
    off_dim_pair correct_value05_arr[2] = { off_dim_pair(32,8),off_dim_pair(80,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value05_arr[0],&correct_value05_arr[0]+2));
    off_dim_pair correct_value06_arr[2] = { off_dim_pair(40,8),off_dim_pair(0,8) };    correct_vals.push_back(sparse_block_tree::value_t(&correct_value06_arr[0],&correct_value06_arr[0]+2));
    off_dim_pair correct_value07_arr[2] = { off_dim_pair(40,8),off_dim_pair(8,8) };    correct_vals.push_back(sparse_block_tree::value_t(&correct_value07_arr[0],&correct_value07_arr[0]+2));
    off_dim_pair correct_value08_arr[2] = { off_dim_pair(48,8),off_dim_pair(24,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value08_arr[0],&correct_value08_arr[0]+2));
    off_dim_pair correct_value09_arr[2] = { off_dim_pair(56,8),off_dim_pair(16,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value09_arr[0],&correct_value09_arr[0]+2));
    off_dim_pair correct_value10_arr[2] = { off_dim_pair(56,8),off_dim_pair(32,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value10_arr[0],&correct_value10_arr[0]+2));
    off_dim_pair correct_value11_arr[2] = { off_dim_pair(64,8),off_dim_pair(40,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value11_arr[0],&correct_value11_arr[0]+2));
    off_dim_pair correct_value12_arr[2] = { off_dim_pair(64,8),off_dim_pair(48,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value12_arr[0],&correct_value12_arr[0]+2));
    off_dim_pair correct_value13_arr[2] = { off_dim_pair(72,8),off_dim_pair(40,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value13_arr[0],&correct_value13_arr[0]+2));
    off_dim_pair correct_value14_arr[2] = { off_dim_pair(72,8),off_dim_pair(48,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value14_arr[0],&correct_value14_arr[0]+2));

    if(!verify_tree(sbt_fused,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::fuse(...) returned incorrect value");
    }
}

//ijk fused to jkl 
void sparse_block_tree_test::test_fuse_3d_3d_multi_index() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_fuse_3d_3d_multi_index";

    three_d_test_f tf = three_d_test_f();
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);
    sparse_block_tree sbt_2(tf.keys,tf.subspaces);

    idx_list lhs_fuse_points(1,1);
    lhs_fuse_points.push_back(2);
    idx_list rhs_fuse_points(1,0);
    rhs_fuse_points.push_back(1);
    sparse_block_tree sbt_fused = sbt_1.fuse(sbt_2,lhs_fuse_points,rhs_fuse_points); 

    //Correct keys
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_key00_arr[4] = {1,2,3,1}; correct_keys.push_back(block_list(&correct_key00_arr[0],&correct_key00_arr[0]+4));
    size_t correct_key01_arr[4] = {2,4,2,2}; correct_keys.push_back(block_list(&correct_key01_arr[0],&correct_key01_arr[0]+4));
    size_t correct_key02_arr[4] = {5,2,6,3}; correct_keys.push_back(block_list(&correct_key02_arr[0],&correct_key02_arr[0]+4));
    size_t correct_key03_arr[4] = {5,2,6,4}; correct_keys.push_back(block_list(&correct_key03_arr[0],&correct_key03_arr[0]+4));
    size_t correct_key04_arr[4] = {7,7,7,7}; correct_keys.push_back(block_list(&correct_key04_arr[0],&correct_key04_arr[0]+4));

    //Correct values
    std::vector< sparse_block_tree::value_t > correct_vals;
    off_dim_pair correct_value00_arr[2] = { off_dim_pair(0,8),off_dim_pair(32,8) };     correct_vals.push_back(sparse_block_tree::value_t(&correct_value00_arr[0],&correct_value00_arr[0]+2));
    off_dim_pair correct_value01_arr[2] = { off_dim_pair(40,8),off_dim_pair(88,8) };    correct_vals.push_back(sparse_block_tree::value_t(&correct_value01_arr[0],&correct_value01_arr[0]+2));
    off_dim_pair correct_value02_arr[2] = { off_dim_pair(128,8),off_dim_pair(56,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value02_arr[0],&correct_value02_arr[0]+2));
    off_dim_pair correct_value03_arr[2] = { off_dim_pair(128,8),off_dim_pair(64,8) };   correct_vals.push_back(sparse_block_tree::value_t(&correct_value03_arr[0],&correct_value03_arr[0]+2));
    off_dim_pair correct_value04_arr[2] = { off_dim_pair(160,8),off_dim_pair(160,8) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value04_arr[0],&correct_value04_arr[0]+2));

    if(!verify_tree(sbt_fused,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::fuse(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_truncate_subspace_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_truncate_subspace_3d()";

    three_d_test_f tf = three_d_test_f();
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);

    idx_pair subspace_bounds(3,5);
    sparse_block_tree sbt_truncated = sbt_1.truncate_subspace(1,subspace_bounds);

    //Correct keys
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_key0_arr[3] = {1,3,1}; correct_keys.push_back(block_list(&correct_key0_arr[0],&correct_key0_arr[0]+3));
    size_t correct_key1_arr[3] = {2,3,1}; correct_keys.push_back(block_list(&correct_key1_arr[0],&correct_key1_arr[0]+3));
    size_t correct_key2_arr[3] = {2,4,2}; correct_keys.push_back(block_list(&correct_key2_arr[0],&correct_key2_arr[0]+3));
    size_t correct_key3_arr[3] = {2,4,5}; correct_keys.push_back(block_list(&correct_key3_arr[0],&correct_key3_arr[0]+3));
    size_t correct_key4_arr[3] = {4,3,5}; correct_keys.push_back(block_list(&correct_key4_arr[0],&correct_key4_arr[0]+3));
    size_t correct_key5_arr[3] = {4,3,6}; correct_keys.push_back(block_list(&correct_key5_arr[0],&correct_key5_arr[0]+3));
    size_t correct_key6_arr[3] = {4,3,7}; correct_keys.push_back(block_list(&correct_key6_arr[0],&correct_key6_arr[0]+3));
    size_t correct_key7_arr[3] = {7,4,5}; correct_keys.push_back(block_list(&correct_key7_arr[0],&correct_key7_arr[0]+3));
    size_t correct_key8_arr[3] = {7,4,6}; correct_keys.push_back(block_list(&correct_key8_arr[0],&correct_key8_arr[0]+3));

    //Correct values
    std::vector< sparse_block_tree::value_t > correct_vals;
    off_dim_pair correct_value0_arr[1] = { off_dim_pair(16,8) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value0_arr[0],&correct_value0_arr[0]+1));
    off_dim_pair correct_value1_arr[1] = { off_dim_pair(32,8) };  correct_vals.push_back(sparse_block_tree::value_t(&correct_value1_arr[0],&correct_value1_arr[0]+1));
    off_dim_pair correct_value2_arr[1] = { off_dim_pair(40,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value2_arr[0],&correct_value2_arr[0]+1));
    off_dim_pair correct_value3_arr[1] = { off_dim_pair(48,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value3_arr[0],&correct_value3_arr[0]+1));
    off_dim_pair correct_value4_arr[1] = { off_dim_pair(96,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value4_arr[0],&correct_value4_arr[0]+1));
    off_dim_pair correct_value5_arr[1] = { off_dim_pair(104,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value5_arr[0],&correct_value5_arr[0]+1));
    off_dim_pair correct_value6_arr[1] = { off_dim_pair(112,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value6_arr[0],&correct_value6_arr[0]+1));
    off_dim_pair correct_value7_arr[1] = { off_dim_pair(144,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value7_arr[0],&correct_value7_arr[0]+1));
    off_dim_pair correct_value8_arr[1] = { off_dim_pair(152,8) }; correct_vals.push_back(sparse_block_tree::value_t(&correct_value8_arr[0],&correct_value8_arr[0]+1));

    if(!verify_tree(sbt_truncated,correct_keys,correct_vals))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_treeL::truncate_subspace(...) returned incorrect value");
    }
}

void sparse_block_tree_test::test_insert_subspace_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_block_tree_test::test_insert_subspace_3d()";

    three_d_test_f tf = three_d_test_f();
    sparse_block_tree sbt_1(tf.keys,tf.subspaces);

    sparse_bispace<1> subspace(4);
    subspace.split(std::vector<size_t>(1,2));
    sparse_block_tree sbt_inserted = sbt_1.insert_subspace(2,subspace);

    //Correct keys
    std::vector< std::vector<size_t> > correct_keys;
    size_t correct_key00_arr[4] = {1,2,0,3}; correct_keys.push_back(block_list(&correct_key00_arr[0],&correct_key00_arr[0]+4));
    size_t correct_key01_arr[4] = {1,2,0,7}; correct_keys.push_back(block_list(&correct_key01_arr[0],&correct_key01_arr[0]+4));
    size_t correct_key02_arr[4] = {1,2,1,3}; correct_keys.push_back(block_list(&correct_key02_arr[0],&correct_key02_arr[0]+4));
    size_t correct_key03_arr[4] = {1,2,1,7}; correct_keys.push_back(block_list(&correct_key03_arr[0],&correct_key03_arr[0]+4));
    size_t correct_key04_arr[4] = {1,3,0,1}; correct_keys.push_back(block_list(&correct_key04_arr[0],&correct_key04_arr[0]+4));
    size_t correct_key05_arr[4] = {1,3,1,1}; correct_keys.push_back(block_list(&correct_key05_arr[0],&correct_key05_arr[0]+4));
    size_t correct_key06_arr[4] = {1,5,0,9}; correct_keys.push_back(block_list(&correct_key06_arr[0],&correct_key06_arr[0]+4));
    size_t correct_key07_arr[4] = {1,5,1,9}; correct_keys.push_back(block_list(&correct_key07_arr[0],&correct_key07_arr[0]+4));
    size_t correct_key08_arr[4] = {2,3,0,1}; correct_keys.push_back(block_list(&correct_key08_arr[0],&correct_key08_arr[0]+4));
    size_t correct_key09_arr[4] = {2,3,1,1}; correct_keys.push_back(block_list(&correct_key09_arr[0],&correct_key09_arr[0]+4));
    size_t correct_key10_arr[4] = {2,4,0,2}; correct_keys.push_back(block_list(&correct_key10_arr[0],&correct_key10_arr[0]+4));
    size_t correct_key11_arr[4] = {2,4,0,5}; correct_keys.push_back(block_list(&correct_key11_arr[0],&correct_key11_arr[0]+4));
    size_t correct_key12_arr[4] = {2,4,1,2}; correct_keys.push_back(block_list(&correct_key12_arr[0],&correct_key12_arr[0]+4));
    size_t correct_key13_arr[4] = {2,4,1,5}; correct_keys.push_back(block_list(&correct_key13_arr[0],&correct_key13_arr[0]+4));
    size_t correct_key14_arr[4] = {2,6,0,3}; correct_keys.push_back(block_list(&correct_key14_arr[0],&correct_key14_arr[0]+4));
    size_t correct_key15_arr[4] = {2,6,0,4}; correct_keys.push_back(block_list(&correct_key15_arr[0],&correct_key15_arr[0]+4));
    size_t correct_key16_arr[4] = {2,6,1,3}; correct_keys.push_back(block_list(&correct_key16_arr[0],&correct_key16_arr[0]+4));
    size_t correct_key17_arr[4] = {2,6,1,4}; correct_keys.push_back(block_list(&correct_key17_arr[0],&correct_key17_arr[0]+4));
    size_t correct_key18_arr[4] = {4,1,0,4}; correct_keys.push_back(block_list(&correct_key18_arr[0],&correct_key18_arr[0]+4));
    size_t correct_key19_arr[4] = {4,1,0,7}; correct_keys.push_back(block_list(&correct_key19_arr[0],&correct_key19_arr[0]+4));
    size_t correct_key20_arr[4] = {4,1,1,4}; correct_keys.push_back(block_list(&correct_key20_arr[0],&correct_key20_arr[0]+4));
    size_t correct_key21_arr[4] = {4,1,1,7}; correct_keys.push_back(block_list(&correct_key21_arr[0],&correct_key21_arr[0]+4));
    size_t correct_key22_arr[4] = {4,2,0,2}; correct_keys.push_back(block_list(&correct_key22_arr[0],&correct_key22_arr[0]+4));
    size_t correct_key23_arr[4] = {4,2,1,2}; correct_keys.push_back(block_list(&correct_key23_arr[0],&correct_key23_arr[0]+4));
    size_t correct_key24_arr[4] = {4,3,0,5}; correct_keys.push_back(block_list(&correct_key24_arr[0],&correct_key24_arr[0]+4));
    size_t correct_key25_arr[4] = {4,3,0,6}; correct_keys.push_back(block_list(&correct_key25_arr[0],&correct_key25_arr[0]+4));
    size_t correct_key26_arr[4] = {4,3,0,7}; correct_keys.push_back(block_list(&correct_key26_arr[0],&correct_key26_arr[0]+4));
    size_t correct_key27_arr[4] = {4,3,1,5}; correct_keys.push_back(block_list(&correct_key27_arr[0],&correct_key27_arr[0]+4));
    size_t correct_key28_arr[4] = {4,3,1,6}; correct_keys.push_back(block_list(&correct_key28_arr[0],&correct_key28_arr[0]+4));
    size_t correct_key29_arr[4] = {4,3,1,7}; correct_keys.push_back(block_list(&correct_key29_arr[0],&correct_key29_arr[0]+4));
    size_t correct_key30_arr[4] = {5,1,0,4}; correct_keys.push_back(block_list(&correct_key30_arr[0],&correct_key30_arr[0]+4));
    size_t correct_key31_arr[4] = {5,1,1,4}; correct_keys.push_back(block_list(&correct_key31_arr[0],&correct_key31_arr[0]+4));
    size_t correct_key32_arr[4] = {5,2,0,6}; correct_keys.push_back(block_list(&correct_key32_arr[0],&correct_key32_arr[0]+4));
    size_t correct_key33_arr[4] = {5,2,0,7}; correct_keys.push_back(block_list(&correct_key33_arr[0],&correct_key33_arr[0]+4));
    size_t correct_key34_arr[4] = {5,2,1,6}; correct_keys.push_back(block_list(&correct_key34_arr[0],&correct_key34_arr[0]+4));
    size_t correct_key35_arr[4] = {5,2,1,7}; correct_keys.push_back(block_list(&correct_key35_arr[0],&correct_key35_arr[0]+4));
    size_t correct_key36_arr[4] = {7,4,0,5}; correct_keys.push_back(block_list(&correct_key36_arr[0],&correct_key36_arr[0]+4));
    size_t correct_key37_arr[4] = {7,4,0,6}; correct_keys.push_back(block_list(&correct_key37_arr[0],&correct_key37_arr[0]+4));
    size_t correct_key38_arr[4] = {7,4,1,5}; correct_keys.push_back(block_list(&correct_key38_arr[0],&correct_key38_arr[0]+4));
    size_t correct_key39_arr[4] = {7,4,1,6}; correct_keys.push_back(block_list(&correct_key39_arr[0],&correct_key39_arr[0]+4));
    size_t correct_key40_arr[4] = {7,7,0,7}; correct_keys.push_back(block_list(&correct_key40_arr[0],&correct_key40_arr[0]+4));
    size_t correct_key41_arr[4] = {7,7,1,7}; correct_keys.push_back(block_list(&correct_key41_arr[0],&correct_key41_arr[0]+4));

    size_t m = 0;
    for(sparse_block_tree::iterator it = sbt_inserted.begin(); it != sbt_inserted.end(); ++it)
    {
        if(it.key() != correct_keys[m])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_block_tree::insert_subspace(...) returned incorrect value");
        }
        ++m;
    }
    if(m != correct_keys.size())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_block_tree::insert_subspace(...) returned incorrect value");
    }
}

} // namespace libtensor

