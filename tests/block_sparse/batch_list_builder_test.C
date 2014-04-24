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
//and chooses that batches such that no batched bispace exceeds the memory limit
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
        for(size_t i = 0; i < batch_list.size(); ++i)
        {
            cout << batch_list[i].first << "," << batch_list[i].second << "\n";
        }
        fail_test(test_name,__FILE__,__LINE__,
                "batch_list_builder::get_batch_list(...) did not return correct value for dense dense case");
    }
}



} // namespace libtensor
