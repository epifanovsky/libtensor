#include "batch_list_builder_test.h"
#include "test_fixtures/index_group_test_f.h"
#include <libtensor/block_sparse/batch_list_builder.h>

using namespace std;

namespace libtensor {

void batch_list_builder_test::perform() throw(libtest::test_exception) 
{
    test_get_batch_list_dense();
    test_get_batch_list_sparse(); 
    test_get_batch_list_dense_dense(); 
    test_get_batch_list_sparse_sparse();
}

void batch_list_builder_test::test_get_batch_list_dense() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list()";

    index_groups_test_f tf = index_groups_test_f();

    /*** BATCHING OVER SUBSPACE 1 - DENSE CASE ***/
    size_t max_n_elem = 388800;
    letter i,j,k,l,m,n,o;
    vector<labeled_bispace> labeled_bispace_group_0(1,labeled_bispace(tf.bispace,i|j|k|l|m|n|o));
    vector< vector<labeled_bispace> > labeled_bispace_groups(1,labeled_bispace_group_0);

    batch_list_builder blb(labeled_bispace_groups,j);
    idx_pair_list batch_list = blb.get_batch_list(max_n_elem);

    idx_pair_list correct_batch_list(1,idx_pair(0,3));
    correct_batch_list.push_back(idx_pair(3,5));
    if(batch_list != correct_batch_list)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for batching over subspace 1 for 1 bispace");
    }
}

void batch_list_builder_test::test_get_batch_list_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list_sparse()";

    index_groups_test_f tf = index_groups_test_f();

    /*** BATCHING OVER SUBSPACE 2 - SPARSE CASE ***/
    size_t max_n_elem = 259200;
    letter i,j,k,l,m,n,o;
    vector<labeled_bispace> labeled_bispace_group_0(1,labeled_bispace(tf.bispace,i|j|k|l|m|n|o));
    vector< vector<labeled_bispace> > labeled_bispace_groups(1,labeled_bispace_group_0);

    batch_list_builder blb(labeled_bispace_groups,k);
    idx_pair_list batch_list = blb.get_batch_list(max_n_elem);

    idx_pair_list correct_batch_list(1,idx_pair(0,3));
    correct_batch_list.push_back(idx_pair(3,4));
    correct_batch_list.push_back(idx_pair(4,5));
    correct_batch_list.push_back(idx_pair(5,6));

    if(batch_list != correct_batch_list)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for batching over subspace 2 for 1 bispace");
    }
}

//Show that batching correctly accounts for different inner sizes of multiple batched bispaces  
void batch_list_builder_test::test_get_batch_list_dense_dense() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list_dense_dense()";

    index_groups_test_f tf = index_groups_test_f();
    sparse_bispace<1> spb_0 = tf.bispace[0];
    sparse_bispace<1> spb_1 = tf.bispace[2];
    letter i,j;
    vector<labeled_bispace> labeled_bispace_group_0(1,labeled_bispace(spb_0|spb_0,i|j));
    labeled_bispace_group_0.push_back(labeled_bispace(spb_0|spb_1,i|j));
    vector< vector<labeled_bispace> > labeled_bispace_groups(1,labeled_bispace_group_0);

    batch_list_builder blb(labeled_bispace_groups,i);
    size_t max_n_elem = 112;
    idx_pair_list batch_list = blb.get_batch_list(max_n_elem);

    idx_pair_list correct_batch_list(1,idx_pair(0,2));
    correct_batch_list.push_back(idx_pair(2,4));
    correct_batch_list.push_back(idx_pair(4,5));

    if(batch_list != correct_batch_list)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for dense dense case");
    }
}

//Show that batching can correctly reconcile different sparsity patterns
void batch_list_builder_test::test_get_batch_list_sparse_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list_sparse_sparse()";

    sparse_bispace<1> spb_0(12);
    vector<size_t> split_points_0;
    for(size_t i = 2; i < spb_0.get_dim(); i += 2)
    {
        split_points_0.push_back(i);
    }
    spb_0.split(split_points_0);

    //Make A - we will force to be permuted for batch for extra complexity
    size_t key_0_arr_A[2] = {0,2};
    size_t key_1_arr_A[2] = {0,3};
    size_t key_2_arr_A[2] = {0,4};
    vector< sequence<2,size_t> > sig_blocks_A(3);
    for(size_t i = 0; i < 2; ++i) sig_blocks_A[0][i] = key_0_arr_A[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_A[1][i] = key_1_arr_A[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_A[2][i] = key_2_arr_A[i];

    sparse_bispace<2> spb_A = spb_0 % spb_0 << sig_blocks_A;

    //Make B 
    size_t key_0_arr_B[2] = {1,0};
    size_t key_1_arr_B[2] = {2,0};
    size_t key_2_arr_B[2] = {5,0};
    vector< sequence<2,size_t> > sig_blocks_B(3);
    for(size_t i = 0; i < 2; ++i) sig_blocks_B[0][i] = key_0_arr_B[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_B[1][i] = key_1_arr_B[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_B[2][i] = key_2_arr_B[i];

    sparse_bispace<2> spb_B = spb_0 % spb_0 << sig_blocks_B;

    letter i,j;
    vector<labeled_bispace> labeled_bispace_group_0(1,labeled_bispace(spb_A,j|i));
    labeled_bispace_group_0.push_back(labeled_bispace(spb_B,i|j));
    vector< vector<labeled_bispace> > labeled_bispace_groups(1,labeled_bispace_group_0);

    batch_list_builder blb(labeled_bispace_groups,i);
    size_t max_n_elem = 8;
    idx_pair_list batch_list = blb.get_batch_list(max_n_elem);

    std::vector<idx_pair> correct_batch_list(1,idx_pair(0,2));
    correct_batch_list.push_back(idx_pair(2,3));
    correct_batch_list.push_back(idx_pair(3,5));
    correct_batch_list.push_back(idx_pair(5,6));

    if(batch_list != correct_batch_list)
    {
        for(size_t i = 0; i < batch_list.size(); ++i)
        {
            cout << batch_list[i].first << "," << batch_list[i].second << "\n";
        }
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for sparse sparse case");
    }
}



} // namespace libtensor
