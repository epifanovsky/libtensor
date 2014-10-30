#include "sparse_bispace_impl_test.h"
#include <libtensor/block_sparse/sparse_bispace_impl.h>

using namespace std;

namespace libtensor {

void sparse_bispace_impl_test::perform() throw(libtest::test_exception)
{
#if 0
    test_equality_2d();
    test_equality_2d_sparse();

    test_permute_2d_10();
    test_permute_3d_dense_sparse_021();
    test_permute_3d_non_contiguous_sparsity();
    test_permute_3d_fully_sparse_210();

    test_contract_3d_dense();
#endif
}

#if 0
/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_impl_test::test_equality_2d() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_equality_2d";

    size_t sp_0[2] = {1,3};  
    subspace sub_0(5,idx_list(sp_0,sp_0+2));

    size_t sp_1[2] = {2,5};  
    subspace sub_1(6,idx_list(sp_1,sp_1+2));

    sparse_bispace_impl spb_0(sub_0,sub_1);
    sparse_bispace_impl spb_1(sub_0,sub_1);

    if(!(spb_0 == spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned false");
    }
}

void sparse_bispace_impl_test::test_equality_2d_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_equality_2d_sparse()";

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    size_t sp_0[3] = {2,5,9};
    vector<subspace> subspaces(2,subspace(11,idx_list(sp_0,sp_0+3)));

    //Specify different sets of significant blocks
    vector< sequence<2,size_t> > sig_blocks_0(4);
    sig_blocks_0[0][0] = 0; 
    sig_blocks_0[0][1] = 1;
    sig_blocks_0[1][0] = 1;
    sig_blocks_0[1][1] = 2;
    sig_blocks_0[2][0] = 2;
    sig_blocks_0[2][1] = 3;
    sig_blocks_0[3][0] = 3;
    sig_blocks_0[3][1] = 2;

    vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 1;//! Changed this one value
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    sparse_bispace_impl two_d_0(subspaces,sparse_block_tree(sig_blocks_0,subspaces));
    sparse_bispace_impl two_d_1(subspaces,sparse_block_tree(sig_blocks_0,subspaces));
    sparse_bispace_impl two_d_2(subspaces,sparse_block_tree(sig_blocks_1,subspaces));

    if(two_d_0 != two_d_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }

    if(two_d_0 == two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_2d_10() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_2d_10()";

    size_t sp_0[2] = {1,3};  
    subspace sub_0(5,idx_list(sp_0,sp_0+2));

    size_t sp_1[2] = {2,5};  
    subspace sub_1(6,idx_list(sp_1,sp_1+2));

    sparse_bispace_impl spb_0(sub_0,sub_1);
    sparse_bispace_impl spb_1(sub_1,sub_0);

    runtime_permutation perm(2);
    perm.permute(0,1);
    if(spb_0.permute(perm) != spb_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");

    }
}

void sparse_bispace_impl_test::test_permute_3d_dense_sparse_021() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_3d_dense_sparse_021()";

    size_t sp_0[3] = {2,5,9};
    subspace sub_0(11,idx_list(sp_0,sp_0+3));

    size_t sp_1[2] = {2,5};
    subspace sub_1(9,idx_list(sp_1,sp_1+2));

    vector< sequence<2,size_t> > sig_blocks(4);
    size_t orig_key0_arr[2] = {0,1}; for(size_t i = 0; i < 2; ++i) sig_blocks[0][i] = orig_key0_arr[i];
    size_t orig_key1_arr[2] = {1,2}; for(size_t i = 0; i < 2; ++i) sig_blocks[1][i] = orig_key1_arr[i];
    size_t orig_key2_arr[2] = {2,3}; for(size_t i = 0; i < 2; ++i) sig_blocks[2][i] = orig_key2_arr[i];
    size_t orig_key3_arr[2] = {3,2}; for(size_t i = 0; i < 2; ++i) sig_blocks[3][i] = orig_key3_arr[i];

    vector<subspace> sparse_subspaces(2,sub_0);
    sparse_bispace_impl sparse_grp(sparse_subspaces,sparse_block_tree(sig_blocks,sparse_subspaces));
    sparse_bispace_impl three_d(sub_1,sparse_grp);
    
    //Construct the benchmark permuted space
    runtime_permutation perm(3);
    perm.permute(1,2);
    vector< sequence<2,size_t> > permuted_sig_blocks(4);
    size_t permuted_key0_arr[2] = {1,0}; for(size_t i = 0; i < 2; ++i) permuted_sig_blocks[0][i] = permuted_key0_arr[i];
    size_t permuted_key1_arr[2] = {2,1}; for(size_t i = 0; i < 2; ++i) permuted_sig_blocks[1][i] = permuted_key1_arr[i];
    size_t permuted_key2_arr[2] = {2,3}; for(size_t i = 0; i < 2; ++i) permuted_sig_blocks[2][i] = permuted_key2_arr[i];
    size_t permuted_key3_arr[2] = {3,2}; for(size_t i = 0; i < 2; ++i) permuted_sig_blocks[3][i] = permuted_key3_arr[i];

    sparse_bispace_impl permuted_sparse_grp(sparse_subspaces,sparse_block_tree(permuted_sig_blocks,sparse_subspaces));
    sparse_bispace_impl correct_three_d(sub_1,permuted_sparse_grp);

    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_3d_non_contiguous_sparsity() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_3d_non_contiguous_sparsity()";

    size_t sp_0[3] = {2,5,9};
    subspace sub_0(11,idx_list(sp_0,sp_0+3));

    size_t sp_1[2] = {2,5};
    subspace sub_1(9,idx_list(sp_1,sp_1+2));

    std::vector< sequence<2,size_t> > sig_blocks(5);
    size_t orig_key0_arr[2] = {0,1}; for(size_t i = 0; i < 2; ++i) sig_blocks[0][i] = orig_key0_arr[i];
    size_t orig_key1_arr[2] = {1,2}; for(size_t i = 0; i < 2; ++i) sig_blocks[1][i] = orig_key1_arr[i];
    size_t orig_key2_arr[2] = {2,2}; for(size_t i = 0; i < 2; ++i) sig_blocks[2][i] = orig_key2_arr[i];
    size_t orig_key3_arr[2] = {2,3}; for(size_t i = 0; i < 2; ++i) sig_blocks[3][i] = orig_key3_arr[i];
    size_t orig_key4_arr[2] = {3,1}; for(size_t i = 0; i < 2; ++i) sig_blocks[4][i] = orig_key4_arr[i];

    vector<subspace> sparse_subspaces(2,sub_0);
    sparse_bispace_impl sparse_grp(sparse_subspaces,sparse_block_tree(sig_blocks,sparse_subspaces));
    sparse_bispace_impl three_d(sub_1,sparse_grp);
    
    //Construct the benchmark permuted space
    runtime_permutation perm(3);
    perm.permute(0,2);
    perm.permute(1,2);
    std::vector< sequence<3,size_t> > permuted_sig_blocks(15);
    size_t permuted_key00_arr[3] = {1,0,0}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[0][i] =  permuted_key00_arr[i];
    size_t permuted_key01_arr[3] = {1,0,3}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[1][i] =  permuted_key01_arr[i];
    size_t permuted_key02_arr[3] = {1,1,0}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[2][i] =  permuted_key02_arr[i];
    size_t permuted_key03_arr[3] = {1,1,3}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[3][i] =  permuted_key03_arr[i];
    size_t permuted_key04_arr[3] = {1,2,0}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[4][i] =  permuted_key04_arr[i];
    size_t permuted_key05_arr[3] = {1,2,3}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[5][i] =  permuted_key05_arr[i];
    size_t permuted_key06_arr[3] = {2,0,1}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[6][i] =  permuted_key06_arr[i];
    size_t permuted_key07_arr[3] = {2,0,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[7][i] =  permuted_key07_arr[i];
    size_t permuted_key08_arr[3] = {2,1,1}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[8][i] =  permuted_key08_arr[i];
    size_t permuted_key09_arr[3] = {2,1,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[9][i] =  permuted_key09_arr[i];
    size_t permuted_key10_arr[3] = {2,2,1}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[10][i] = permuted_key10_arr[i];
    size_t permuted_key11_arr[3] = {2,2,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[11][i] = permuted_key11_arr[i];
    size_t permuted_key12_arr[3] = {3,0,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[12][i] = permuted_key12_arr[i];
    size_t permuted_key13_arr[3] = {3,1,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[13][i] = permuted_key13_arr[i];
    size_t permuted_key14_arr[3] = {3,2,2}; for(size_t i = 0; i < 3; ++i) permuted_sig_blocks[14][i] = permuted_key14_arr[i];

    vector<subspace> permuted_sparse_subspaces(1,sub_0);
    permuted_sparse_subspaces.push_back(sub_1);
    permuted_sparse_subspaces.push_back(sub_0);
    sparse_bispace_impl correct_three_d(permuted_sparse_subspaces,sparse_block_tree( permuted_sig_blocks,permuted_sparse_subspaces));

    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_3d_fully_sparse_210() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_3d_fully_sparse_210()";

    size_t sp_0[3] = {2,5,9};
    subspace sub_0(11,idx_list(sp_0,sp_0+3));

    size_t sp_1[2] = {2,5};
    subspace sub_1(9,idx_list(sp_1,sp_1+2));

    std::vector< sequence<3,size_t> > sig_blocks(5);

    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 0;
    sig_blocks[0][2] = 2;
    sig_blocks[1][0] = 0; 
    sig_blocks[1][1] = 0;
    sig_blocks[1][2] = 3;
    sig_blocks[2][0] = 1;
    sig_blocks[2][1] = 2;
    sig_blocks[2][2] = 2;
    sig_blocks[3][0] = 1;
    sig_blocks[3][1] = 3;
    sig_blocks[3][2] = 1;
    sig_blocks[4][0] = 2;
    sig_blocks[4][1] = 0;
    sig_blocks[4][2] = 1;

    vector<subspace> subspaces(1,sub_1);
    subspaces.push_back(sub_0);
    subspaces.push_back(sub_0);

    sparse_bispace_impl three_d(subspaces,sparse_block_tree(sig_blocks,subspaces));

    //Make the benchmark
    std::vector< sequence<3,size_t> > permuted_sig_blocks(5);
    permuted_sig_blocks[0][0] = 1;
    permuted_sig_blocks[0][1] = 0;
    permuted_sig_blocks[0][2] = 2;
    permuted_sig_blocks[1][0] = 1;
    permuted_sig_blocks[1][1] = 3;
    permuted_sig_blocks[1][2] = 1;
    permuted_sig_blocks[2][0] = 2; 
    permuted_sig_blocks[2][1] = 0;
    permuted_sig_blocks[2][2] = 0;
    permuted_sig_blocks[3][0] = 2;
    permuted_sig_blocks[3][1] = 2;
    permuted_sig_blocks[3][2] = 1;
    permuted_sig_blocks[4][0] = 3; 
    permuted_sig_blocks[4][1] = 0;
    permuted_sig_blocks[4][2] = 0;
    swap(subspaces[0],subspaces[2]);
    sparse_bispace_impl correct_three_d(subspaces,sparse_block_tree(permuted_sig_blocks,subspaces)); 

    runtime_permutation perm(3);
    perm.permute(0,2);
    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_contract_3d_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_contract_3d_dense()";

#if 0
    size_t sp_0[2] = {2,5};
    subspace sub_0(8,idx_list(sp_0,sp_0+2));

    size_t sp_1[3] = {3,6,8};
    subspace sub_1(9,idx_list(sp_0,sp_0+3));

    size_t sp_2[2] = {4,7};
    subspace sub_2(10,idx_list(sp_0,sp_0+2));

    sparse_bispace_impl three_d(sub_0,sparse_bispace_impl(sub_1,sub_2));
    sparse_bispace_impl two_d = three_d.contract(1);
    sparse_bispace_impl two_d_correct(sub_0,sub_2);

   if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::contract(...) returned incorrect value");
    }
#endif
}
#endif

} // namespace libtensor
