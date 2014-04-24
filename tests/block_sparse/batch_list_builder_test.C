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
    /*test_get_batch_list_2_group_sparse_sparse();*/
}

//Test fixtures
namespace {

//For testing batching when multiple groups of memory-resident tensors must be considered
class multi_group_test_f {
private:
    sparse_bispace<1> spb_0;
    static size_t A_arr[3][2];
    static size_t B_arr[3][2];
    static size_t C_arr[4][2];
    static size_t D_arr[6][2];
    static sparse_bispace<1> get_spb_0()
    {
        sparse_bispace<1> bs(12);
        vector<size_t> split_points_0;
        for(size_t i = 2; i < bs.get_dim(); i += 2)
        {
            split_points_0.push_back(i);
        }
        bs.split(split_points_0);
        return bs;
    }

    static vector< sequence<2,size_t> > get_sig_blocks(size_t arr[][2],size_t n_entries)
    {
        vector< sequence<2,size_t> > sig_blocks(n_entries);
        for(size_t i = 0; i < n_entries; ++i)
            for(size_t j = 0; j < 2; ++j) sig_blocks[i][j] = arr[i][j];
        return sig_blocks;
    }

public:
    sparse_bispace<2> A;
    sparse_bispace<2> B;
    sparse_bispace<2> C;
    sparse_bispace<2> D;
    multi_group_test_f() : spb_0(get_spb_0()),
                           A(spb_0 % spb_0 << get_sig_blocks(A_arr,sizeof(A_arr)/sizeof(A_arr[0]))),
                           B(spb_0 % spb_0 << get_sig_blocks(B_arr,sizeof(B_arr)/sizeof(B_arr[0]))),
                           C(spb_0 % spb_0 << get_sig_blocks(C_arr,sizeof(C_arr)/sizeof(C_arr[0]))),
                           D(spb_0 % spb_0 << get_sig_blocks(D_arr,sizeof(D_arr)/sizeof(D_arr[0]))) {}
};

size_t multi_group_test_f::A_arr[3][2] = {{0,2},{0,3},{0,4}};
size_t multi_group_test_f::B_arr[3][2] = {{1,0},{2,0},{5,0}};
size_t multi_group_test_f::C_arr[4][2] = {{2,0},{2,1},{3,0},{4,1}};
size_t multi_group_test_f::D_arr[6][2] = {{1,1},{2,1},{3,1},{3,2},{4,4},{5,5}};

} // namespace unnamed

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

    multi_group_test_f tf;
    letter i,j;
    vector<labeled_bispace> labeled_bispace_group_0(1,labeled_bispace(tf.A,j|i));
    labeled_bispace_group_0.push_back(labeled_bispace(tf.B,i|j));
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
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for sparse sparse case");
    }
}

//Now we have two groups of tensors, and no one of them may exceed the element count limit for batch
void batch_list_builder_test::test_get_batch_list_2_group_sparse_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list_2_group_sparse_sparse()";

    multi_group_test_f tf;
    letter i,j;
    vector< vector<labeled_bispace> > labeled_bispace_groups(2);
    labeled_bispace_groups[0].push_back(labeled_bispace(tf.A,j|i));
    labeled_bispace_groups[0].push_back(labeled_bispace(tf.B,i|j));
    labeled_bispace_groups[1].push_back(labeled_bispace(tf.C,i|j));
    labeled_bispace_groups[1].push_back(labeled_bispace(tf.D,i|j));

    batch_list_builder blb(labeled_bispace_groups,i);
    size_t max_n_elem = 12;
    idx_pair_list batch_list = blb.get_batch_list(max_n_elem);

    std::vector<idx_pair> correct_batch_list(1,idx_pair(0,2));
    correct_batch_list.push_back(idx_pair(2,3));
    correct_batch_list.push_back(idx_pair(3,4));
    correct_batch_list.push_back(idx_pair(4,6));

    if(batch_list != correct_batch_list)
    {
        for(size_t i = 0; i < batch_list.size(); ++i)
        {
            cout << batch_list[i].first << "," << batch_list[i].second << "\n";
        }
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for sparse sparse case with two tensor groups");
    }
}



} // namespace libtensor
