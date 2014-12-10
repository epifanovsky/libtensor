#include "sparse_bispace_impl_test.h"
#include <libtensor/block_sparse/sparse_bispace_impl.h>
#include "test_fixtures/index_group_test_f.h"

using namespace std;

namespace libtensor {

void sparse_bispace_impl_test::perform() throw(libtest::test_exception)
{
    test_equality_2d();
    test_equality_2d_sparse();

    test_permute_2d_10();
    test_permute_3d_fully_sparse_210();
    test_permute_3d_non_contiguous_sparsity();
    test_permute_5d_sd_swap();
    test_permute_5d_sd_interleave();

    test_contract_3d_dense();
    test_contract_3d_sparse();
    test_contract_3d_sparsity_destroyed();

    test_get_n_ig();
    test_get_ig_offset();
    test_get_ig_order();
    test_get_ig_dim();
    test_get_ig_containing_subspace();
}

/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_impl_test::test_equality_2d() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_impl_test::test_equality_2d";

    size_t sp_0[2] = {1,3};  
    vector<subspace> subspaces(1,subspace(5,idx_list(sp_0,sp_0+2)));

    size_t sp_1[2] = {2,5};  
    subspaces.push_back(subspace(6,idx_list(sp_1,sp_1+2)));

    sparse_bispace_impl spb_0(subspaces);
    sparse_bispace_impl spb_1(subspaces);

    if(!(spb_0 == spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned false");
    }
}

void sparse_bispace_impl_test::test_equality_2d_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_equality_2d_sparse()";

    size_t sp_0[3] = {2,5,9};
    vector<subspace> subspaces(2,subspace(11,idx_list(sp_0,sp_0+3)));

    size_t keys_arr_0[4][2] = {{0,1},
                               {1,2},
                               {2,3},
                               {3,2}};
    vector<idx_list> keys_0;
    for(size_t key_idx = 0; key_idx < 4; ++key_idx)
        keys_0.push_back(idx_list(keys_arr_0[key_idx],keys_arr_0[key_idx]+2));

    vector<sparsity_data> group_sd(1,sparsity_data(2,keys_0));
    sparse_bispace_impl two_d_0(subspaces,group_sd,idx_list(1,0));
    sparse_bispace_impl two_d_1(subspaces,group_sd,idx_list(1,0));

    if(two_d_0 != two_d_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }

    vector<idx_list> keys_1(keys_0);
    keys_1[2][1] = 1;
    group_sd[0] = sparsity_data(2,keys_1);
    sparse_bispace_impl two_d_2(subspaces,group_sd,idx_list(1,0));
    if((two_d_0 == two_d_2) || (two_d_1 == two_d_2))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_2d_10() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_2d_10()";

    size_t sp_0[2] = {1,3};  
    vector<subspace> subspaces(1,subspace(5,idx_list(sp_0,sp_0+2)));

    size_t sp_1[2] = {2,5};  
    subspaces.push_back(subspace(6,idx_list(sp_1,sp_1+2)));

    sparse_bispace_impl spb_0(subspaces);
    swap(subspaces[0],subspaces[1]);
    sparse_bispace_impl spb_1(subspaces);

    runtime_permutation perm(2);
    perm.permute(0,1);
    if(spb_0.permute(perm) != spb_1)
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

    size_t keys_arr[5][3] = {{0,0,2},
                             {0,0,3},
                             {1,2,2},
                             {1,3,1},
                             {2,0,1}};

    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+3));

    vector<subspace> subspaces(1,sub_1);
    subspaces.push_back(sub_0);
    subspaces.push_back(sub_0);

    vector<sparsity_data> group_sd(1,sparsity_data(3,keys));
    sparse_bispace_impl three_d(subspaces,group_sd,idx_list(1,0));

    size_t p_keys_arr[5][3] = {{1,0,2},
                               {1,3,1},
                               {2,0,0},
                               {2,2,1},
                               {3,0,0}};

    vector<idx_list> p_keys;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        p_keys.push_back(idx_list(p_keys_arr[key_idx],p_keys_arr[key_idx]+3));

    swap(subspaces[0],subspaces[2]);
    group_sd[0] = sparsity_data(3,p_keys);
    sparse_bispace_impl correct_three_d(subspaces,group_sd,idx_list(1,0)); 

    runtime_permutation perm(3);
    perm.permute(0,2);
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

    size_t keys_arr[5][2] = {{0,1},
                             {1,2},
                             {2,2},
                             {2,3},
                             {3,1}};

    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+2));

    vector<subspace> subspaces(2,sub_0);
    subspaces.insert(subspaces.begin(),sub_1);

    vector<sparsity_data> group_sd(1,sparsity_data(2,keys));
    sparse_bispace_impl three_d(subspaces,group_sd,idx_list(1,1));
    
    //Construct the benchmark permuted space
    runtime_permutation perm(3);
    perm.permute(0,2);
    perm.permute(1,2);

    size_t p_keys_arr[15][3] = {{1,0,0}, 
                                {1,0,3},
                                {1,1,0},
                                {1,1,3},
                                {1,2,0},
                                {1,2,3},
                                {2,0,1},
                                {2,0,2},
                                {2,1,1},
                                {2,1,2},
                                {2,2,1},
                                {2,2,2},
                                {3,0,2},
                                {3,1,2},
                                {3,2,2}};

    vector<idx_list> p_keys;
    for(size_t key_idx = 0; key_idx < 15; ++key_idx)
        p_keys.push_back(idx_list(p_keys_arr[key_idx],p_keys_arr[key_idx]+3));

    vector<subspace> p_subspaces(subspaces);
    swap(p_subspaces[0],p_subspaces[1]);
    group_sd[0] = sparsity_data(3,p_keys);
    sparse_bispace_impl correct_three_d(p_subspaces,group_sd,idx_list(1,0));

    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_5d_sd_swap() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_5d_sd_swap()";

    size_t sp_0[3] = {2,5,9};
    subspace sub_0(11,idx_list(sp_0,sp_0+3));

    size_t sp_1[2] = {2,5};
    subspace sub_1(9,idx_list(sp_1,sp_1+2));

    size_t keys_arr_0[5][2] = {{0,1},
                               {1,2},
                               {2,2},
                               {2,3},
                               {3,1}};

    vector<idx_list> keys_0;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys_0.push_back(idx_list(keys_arr_0[key_idx],keys_arr_0[key_idx]+2));

    size_t keys_arr_1[4][2] = {{0,1},
                               {0,2},
                               {1,1},
                               {2,1}};

    vector<idx_list> keys_1;
    for(size_t key_idx = 0; key_idx < 4; ++key_idx)
        keys_1.push_back(idx_list(keys_arr_1[key_idx],keys_arr_1[key_idx]+2));

    vector<subspace> subspaces(2,sub_0);
    subspaces.push_back(sub_1);
    subspaces.push_back(sub_1);
    subspaces.push_back(sub_1);

    vector<sparsity_data> group_sd(1,sparsity_data(2,keys_0));
    group_sd.push_back(sparsity_data(2,keys_1));
    idx_list group_offsets(1,0);
    group_offsets.push_back(3);
    sparse_bispace_impl five_d(subspaces,group_sd,group_offsets);
    
    runtime_permutation perm(5);
    perm.permute(0,4);
    perm.permute(1,3);
    perm.permute(2,3);

    //Trees swap order, and dense subspace gets inserted
    size_t p_keys_arr_0[4][2] = {{1,0},
                                 {1,1},
                                 {1,2},
                                 {2,0}};

    vector<idx_list> p_keys_0;
    for(size_t key_idx = 0; key_idx < 4; ++key_idx)
        p_keys_0.push_back(idx_list(p_keys_arr_0[key_idx],p_keys_arr_0[key_idx]+2));

    size_t p_keys_arr_1[15][3] = {{1,0,0}, 
                                  {1,0,3},
                                  {1,1,0},
                                  {1,1,3},
                                  {1,2,0},
                                  {1,2,3},
                                  {2,0,1},
                                  {2,0,2},
                                  {2,1,1},
                                  {2,1,2},
                                  {2,2,1},
                                  {2,2,2},
                                  {3,0,2},
                                  {3,1,2},
                                  {3,2,2}};

    vector<idx_list> p_keys_1;
    for(size_t key_idx = 0; key_idx < 15; ++key_idx)
        p_keys_1.push_back(idx_list(p_keys_arr_1[key_idx],p_keys_arr_1[key_idx]+3));

    vector<subspace> p_subspaces(subspaces);
    swap(p_subspaces[0],p_subspaces[4]);
    swap(p_subspaces[1],p_subspaces[3]);
    swap(p_subspaces[2],p_subspaces[3]);
    vector<sparsity_data> p_group_sd(1,sparsity_data(2,p_keys_0));
    p_group_sd.push_back(sparsity_data(3,p_keys_1));
    idx_list p_group_offsets(1,0);
    p_group_offsets.push_back(2);
    sparse_bispace_impl correct_five_d(p_subspaces,p_group_sd,p_group_offsets);

    if(five_d.permute(perm) != correct_five_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_permute_5d_sd_interleave() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_5d_sd_interleave()";

    size_t sp_0[3] = {2,5,9};
    subspace sub_0(11,idx_list(sp_0,sp_0+3));

    size_t sp_1[2] = {2,5};
    subspace sub_1(9,idx_list(sp_1,sp_1+2));

    size_t keys_arr_0[5][2] = {{0,1},
                               {1,2},
                               {2,2},
                               {2,3},
                               {3,1}};

    vector<idx_list> keys_0;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys_0.push_back(idx_list(keys_arr_0[key_idx],keys_arr_0[key_idx]+2));

    size_t keys_arr_1[4][2] = {{0,2},
                               {1,1},
                               {2,0},
                               {2,2}};

    vector<idx_list> keys_1;
    for(size_t key_idx = 0; key_idx < 4; ++key_idx)
        keys_1.push_back(idx_list(keys_arr_1[key_idx],keys_arr_1[key_idx]+2));

    vector<subspace> subspaces(2,sub_0);
    subspaces.push_back(sub_1);
    subspaces.push_back(sub_1);
    subspaces.push_back(sub_1);

    vector<sparsity_data> group_sd(1,sparsity_data(2,keys_0));
    group_sd.push_back(sparsity_data(2,keys_1));
    idx_list group_offsets(1,0);
    group_offsets.push_back(3);
    sparse_bispace_impl five_d(subspaces,group_sd,group_offsets);
    
    runtime_permutation perm(5);
    perm.permute(1,3);

    bool threw_exception = false;
    try
    {
        five_d.permute(perm);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) did not throw exception when interleaving trees");
    }
}

void sparse_bispace_impl_test::test_contract_3d_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_contract_3d_dense()";

    size_t sp_0[2] = {2,5};
    vector<subspace> subspaces(1,subspace(8,idx_list(sp_0,sp_0+2)));
    size_t sp_1[3] = {3,6,8};
    subspaces.push_back(subspace(9,idx_list(sp_1,sp_1+3)));
    size_t sp_2[2] = {4,7};
    subspaces.push_back(subspace(10,idx_list(sp_2,sp_2+2)));

    sparse_bispace_impl three_d(subspaces);
    sparse_bispace_impl two_d = three_d.contract(1);
    subspaces.erase(subspaces.begin()+1);
    sparse_bispace_impl two_d_correct(subspaces);

   if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::contract(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_contract_3d_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_contract_3d_sparse()";

    size_t sp_0[2] = {2,5};
    vector<subspace> subspaces(1,subspace(8,idx_list(sp_0,sp_0+2)));
    size_t sp_1[3] = {3,6,8};
    subspaces.push_back(subspace(9,idx_list(sp_1,sp_1+3)));
    size_t sp_2[2] = {4,7};
    subspaces.push_back(subspace(10,idx_list(sp_2,sp_2+2)));

    //Sparsity Info
    size_t keys_arr[14][3] = {{0,0,0},
                              {0,0,2},
                              {0,1,2},
                              {0,2,1},
                              {0,2,2},
                              {0,3,1},
                              {1,0,1},
                              {1,0,2},
                              {1,1,0},
                              {1,2,0},
                              {1,3,1},
                              {2,1,1},
                              {2,2,2},
                              {2,3,0}};

    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 14; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+3));
    vector<sparsity_data> group_sd(1,sparsity_data(3,keys));

    sparse_bispace_impl three_d(subspaces,group_sd,idx_list(1,0));
    sparse_bispace_impl two_d = three_d.contract(2); 

    //Correct result
    size_t c_keys_arr[11][2] = {{0,0},
                                {0,1},
                                {0,2},
                                {0,3},
                                {1,0},
                                {1,1},
                                {1,2},
                                {1,3},
                                {2,1},
                                {2,2},
                                {2,3}};

    vector<idx_list> c_keys;
    for(size_t key_idx = 0; key_idx < 11; ++key_idx)
        c_keys.push_back(idx_list(c_keys_arr[key_idx],c_keys_arr[key_idx]+2));
    group_sd[0] = sparsity_data(2,c_keys);
    subspaces.erase(subspaces.begin()+2);
    sparse_bispace_impl two_d_correct(subspaces,group_sd,idx_list(1,0));

    if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_contract_3d_sparsity_destroyed() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_contract_3d_sparsity_destroyed()";

    size_t sp_0[2] = {2,5};
    vector<subspace> subspaces(1,subspace(8,idx_list(sp_0,sp_0+2)));
    size_t sp_1[3] = {3,6,8};
    subspaces.push_back(subspace(9,idx_list(sp_1,sp_1+3)));
    size_t sp_2[2] = {4,7};
    subspaces.push_back(subspace(10,idx_list(sp_2,sp_2+2)));

    //Sparsity
    size_t keys_arr[5][2] = {{0,1},
                             {0,3},
                             {1,0},
                             {1,2},
                             {2,1}};
    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+2));
    vector<sparsity_data> group_sd(1,sparsity_data(2,keys));

    sparse_bispace_impl three_d(subspaces,group_sd,idx_list(1,0));
    sparse_bispace_impl two_d = three_d.contract(1); 

    subspaces.erase(subspaces.begin()+1);
    sparse_bispace_impl two_d_correct(subspaces);
 
    if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

void sparse_bispace_impl_test::test_get_n_ig() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_get_n_ig()";
    
    index_groups_test_f tf = index_groups_test_f();

    if(tf.bispace.get_n_ig() != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_n_ig(...) did not return correct value");
    }
}

void sparse_bispace_impl_test::test_get_ig_offset() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_get_ig_offset()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_offsets[5] = {0,1,2,4,5};
    for(size_t grp = 0; grp < tf.bispace.get_n_ig(); ++grp)
    {
        if(tf.bispace.get_ig_offset(grp) != correct_offsets[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_ig_offset(...) did not return correct value");
        }
    }
}

void sparse_bispace_impl_test::test_get_ig_order() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_get_ig_order()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_orders[5] = {1,1,2,1,2};
    for(size_t grp = 0; grp < tf.bispace.get_n_ig(); ++grp)
    {
        if(tf.bispace.get_ig_order(grp) != correct_orders[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_ig_order(...) did not return correct value");
        }
    }
}

void sparse_bispace_impl_test::test_get_ig_dim() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_get_ig_dim()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_dims[5] = {10,10,24,10,27};
    for(size_t grp = 0; grp < tf.bispace.get_n_ig(); ++grp)
    {
        if(tf.bispace.get_ig_dim(grp) != correct_dims[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_ig_dim(...) did not return correct value");
        }
    }
}

void sparse_bispace_impl_test::test_get_ig_containing_subspace() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_get_ig_containing_subspace()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_igs[3] = {1,2,4};
    size_t subspaces[3] = {1,3,6};
    for(size_t subspace_idx = 0; subspace_idx < 3; ++subspace_idx)
    {
        if(tf.bispace.get_ig_containing_subspace(subspaces[subspace_idx]) != correct_igs[subspace_idx])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_ig_containing_subspace(...) did not return correct value");
        }
    }
}

} // namespace libtensor
