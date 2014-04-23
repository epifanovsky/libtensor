#include "batch_list_builder_test.h"
#include "test_fixtures/index_group_test_f.h"
#include <libtensor/block_sparse/batch_list_builder.h>

using namespace std;

namespace libtensor {

void batch_list_builder_test::perform() throw(libtest::test_exception) 
{
    test_get_batch_list_dense();
}

void batch_list_builder_test::test_get_batch_list_dense() throw(libtest::test_exception)
{
    static const char *test_name = "batch_list_builder_test::test_get_batch_list()";

    index_groups_test_f tf = index_groups_test_f();

    /*** BATCHING OVER SUBSPACE 1 - DENSE CASE ***/
    size_t max_n_elem = 0.6*tf.bispace.get_nnz();

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



} // namespace libtensor
