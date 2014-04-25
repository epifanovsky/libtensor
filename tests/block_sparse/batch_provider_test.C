#include "batch_provider_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"
#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/block_sparse/batch_provider_new.h>

using namespace std;

namespace libtensor {

using namespace expr;

void batch_provider_test::perform() throw(libtest::test_exception) 
{
    test_permute_3d_sparse_120();
}

void batch_provider_test::test_permute_3d_sparse_120() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_permute_3d_sparse_120()";

    permute_3d_sparse_120_test_f tf;

    sparse_btensor_new<3> A(tf.input_bispace,tf.input_arr,true);
    sparse_btensor_new<3> B(tf.output_bispace);

    node_assign root(3);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    e.add(root_id, node_ident_any_tensor<3,double>(B));

    idx_list perm_entries(1,1);
    perm_entries.push_back(2);
    perm_entries.push_back(0);
    node_transform<double> perm_node(perm_entries, scalar_transf<double>());
    expr_tree::node_id_t perm_node_id = e.add(root_id,perm_node);
    e.add(perm_node_id,node_ident_any_tensor<3,double>(A));

    batch_provider_new<double> bp(e);
    bp.get_batch((double*)B.get_data_ptr());

    sparse_btensor_new<3> B_correct(tf.output_bispace,tf.output_arr,true);

    if(B != B_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for 3d sparse 120 permutation");
    }
}

} // namespace libtensor
