#include "batch_provider_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"
#include "test_fixtures/contract2_test_f.h"
#include "test_fixtures/contract2_permute_nested_test_f.h"
#include "test_fixtures/contract2_subtract2_nested_test_f.h"
#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_contract.h>

#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/block_sparse/direct_sparse_btensor_new.h>
#include <libtensor/block_sparse/batch_provider_new.h>

using namespace std;

namespace libtensor {

using namespace expr;

void batch_provider_test::perform() throw(libtest::test_exception) 
{
    test_permute_3d_sparse_120();
    test_contract2();
    test_contract2_permute_nested(); 
    test_contract2_subtract2_nested();
    test_batchable_subspaces_recursion_addition();
    test_batchable_subspaces_recursion_permutation();
    test_get_batched_subspace_grps();
    test_get_batched_subspace_grps_batchable_subspaces();
}

namespace
{

class fake_batch_provider : public batch_provider_i<double>
{
private:
    size_t m_batchable_subspace;

public:
    fake_batch_provider(size_t batchable_subspace) : m_batchable_subspace(batchable_subspace) {}
    virtual idx_list get_batchable_subspaces() const { return idx_list(1,m_batchable_subspace); }
    virtual void get_batch(double* output_ptr,const bispace_batch_map& bbm = bispace_batch_map()) {}
};

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

    //Finally, check that we can get the full tensor
    batch_provider_new<double> bp(e);
    bp.get_batch((double*)B.get_data_ptr());

    sparse_btensor_new<3> B_correct(tf.output_bispace,tf.output_arr,true);

    if(B != B_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for 3d sparse 120 permutation");
    }
}

void batch_provider_test::test_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_contract2()";

    contract2_test_f tf;

    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<2> C(tf.spb_C);

    node_assign root(2);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    expr_tree::node_id_t C_ident_id = e.add(root_id, node_ident_any_tensor<2,double>(C));

    multimap<size_t,size_t> contr_map;
    contr_map.insert(idx_pair(1,3));
    contr_map.insert(idx_pair(2,4));
    node_contract contr_node(2,contr_map,true);
    expr_tree::node_id_t contr_node_id = e.add(root_id,contr_node);
    e.add(contr_node_id,node_ident_any_tensor<3,double>(A));
    e.add(contr_node_id,node_ident_any_tensor<3,double>(B));


    batch_provider_new<double> bp(e);

    //First test that we can grab whole tensor
    sparse_btensor_new<2> my_C(tf.spb_C);
    bp.get_batch((double*)my_C.get_data_ptr());
    sparse_btensor_new<2> C_correct(tf.spb_C,tf.C_arr,true);

    if(my_C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value contract2 test case");
    }

    //Check batch 0
    double C_batch_arr[12];
    bispace_batch_map bbm; 
    bbm[idx_pair(0,0)] = idx_pair(0,1);
    bp.get_batch(C_batch_arr,bbm);
    for(size_t i =0 ; i < 6; ++i)
    {
        if(C_batch_arr[i] != tf.C_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "batch_provider::get_batch(...) did not return correct batch 0 for contract2 test case");
        }
    }

    //Check batch 1
    bbm[idx_pair(0,0)] = idx_pair(1,2);
    bp.get_batch(C_batch_arr,bbm);
    for(size_t i = 6 ; i < sizeof(tf.C_arr)/sizeof(tf.C_arr[0]); ++i)
    {
        if(C_batch_arr[i - 6] != tf.C_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "batch_provider::get_batch(...) did not return correct batch 1 for contract2 test case");
        }
    }
}

void batch_provider_test::test_contract2_permute_nested() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_contract2_permute_nested()";
    contract2_permute_nested_test_f tf; 

    batch_provider_new<double> bp(tf.tree);
    bp.get_batch((double*)tf.D.get_data_ptr());


    if(tf.D != tf.D_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for contract2_permute_nested test case");
    }
}

void batch_provider_test::test_contract2_subtract2_nested() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_contract2_subtract2_nested()";
    contract2_subtract2_nested_test_f tf; 

    batch_provider_new<double> bp(tf.tree);
    bp.get_batch((double*)tf.E.get_data_ptr());

    if(tf.E != tf.E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for contract2_subtract2_nested test case");
    }

    //Now we see if it can work when we force one of the intermediates (D) to be batched, though the result is not
    idx_pair_list batch_list(1,idx_pair(0,2));
    batch_list.push_back(idx_pair(2,3));
    vector<idx_list> batched_subspace_grps(1,idx_list(1,1));
    batched_subspace_grps.push_back(idx_list(2,1));
    batched_subspace_grps.push_back(idx_list(1,1));

    bp.set_batch_info(batched_subspace_grps,batch_list);
    bp.get_batch((double*)tf.E.get_data_ptr());

    if(tf.E != tf.E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for contract2_subtract2_nested test case");
    }
}

void batch_provider_test::test_batchable_subspaces_recursion_addition() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_batchable_subspace_recursion_addition()";
    sparse_bispace<1> spb_i(5);
    direct_sparse_btensor_new<3> A(spb_i|spb_i|spb_i);
    fake_batch_provider fbp(2);
    A.set_batch_provider(fbp);
    sparse_btensor_new<3> B(A.get_bispace());
    sparse_btensor_new<3> C(A.get_bispace());

    node_assign root(3);
    expr_tree tree(root);
    expr_tree::node_id_t root_id = tree.get_root();
    tree.add(root_id,node_ident_any_tensor<3,double>(C));
    node_add n_add(3);
    expr_tree::node_id_t n_add_id = tree.add(root_id,n_add);
    tree.add(n_add_id,node_ident_any_tensor<3,double>(A));
    tree.add(n_add_id,node_ident_any_tensor<3,double>(B));
    
    batch_provider_new<double> bp(tree);


    if(bp.get_batchable_subspaces() != idx_list(1,2))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batchable_subspaces(...) did not return correct value for recursion case with addition");
    }
}

void batch_provider_test::test_batchable_subspaces_recursion_permutation() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_batchable_subspace_recursion_permutation()";
    sparse_bispace<1> spb_i(5);
    direct_sparse_btensor_new<3> A(spb_i|spb_i|spb_i);
    fake_batch_provider fbp(2);
    A.set_batch_provider(fbp);
    sparse_btensor_new<3> B(A.get_bispace());

    node_assign root(3);
    expr_tree tree(root);
    expr_tree::node_id_t root_id = tree.get_root();
    tree.add(root_id,node_ident_any_tensor<3,double>(B));
    idx_list perm_entries(1,2);
    perm_entries.push_back(1);
    perm_entries.push_back(0);
    node_transform<double> n_tf(perm_entries,scalar_transf<double>());
    expr_tree::node_id_t n_tf_id = tree.add(root_id,n_tf);
    tree.add(n_tf_id,node_ident_any_tensor<3,double>(A));
    
    batch_provider_new<double> bp(tree);


    if(bp.get_batchable_subspaces() != idx_list(1,0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batchable_subspaces(...) did not return correct value for recursion case with permutation");
    }
}

void batch_provider_test::test_get_batched_subspace_grps() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_get_batched_bispace_subspace_grps()";
    
    contract2_subtract2_nested_test_f tf;
    batch_provider_new<double> bp(tf.tree);
    vector<idx_list> batched_subspace_grps;
    bp.get_batched_subspace_grps(batched_subspace_grps);
    vector<idx_list> correct_batched_subspace_grps(1,idx_list(1,0)); //Last contraction, G
    correct_batched_subspace_grps.push_back(idx_list(2,0)); //Subtraction, G and C
    correct_batched_subspace_grps.push_back(idx_list(1,0)); //First contraction, C
    if(batched_subspace_grps != correct_batched_subspace_grps)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batched_bispace_subspace_grps(...) did not return correct value");
    }
}

void batch_provider_test::test_get_batched_subspace_grps_batchable_subspaces() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_get_batched_bispace_subspace_grps_batchable_subspaces()";

    sparse_bispace<1> spb_i(5);
    direct_sparse_btensor_new<3> A(spb_i|spb_i|spb_i);
    fake_batch_provider fbp(0);
    A.set_batch_provider(fbp);
    direct_sparse_btensor_new<3> B(A.get_bispace());
    sparse_btensor_new<3> C(A.get_bispace());

    node_assign root(3);
    expr_tree tree(root);
    expr_tree::node_id_t root_id = tree.get_root();
    tree.add(root_id,node_ident_any_tensor<3,double>(C));
    idx_list perm_entries(1,2);
    perm_entries.push_back(1);
    perm_entries.push_back(0);
    node_transform<double> n_tf_0(perm_entries,scalar_transf<double>());
    expr_tree::node_id_t n_tf_0_id = tree.add(root_id,n_tf_0);
    node_assign n_a_0(3);
    expr_tree::node_id_t  n_a_0_id = tree.add(n_tf_0_id,n_a_0);
    tree.add(n_a_0_id,node_ident_any_tensor<3,double>(B));

    node_transform<double> n_tf_1(perm_entries,scalar_transf<double>());
    expr_tree::node_id_t n_tf_1_id = tree.add(n_a_0_id,n_tf_1);
    tree.add(n_tf_1_id,node_ident_any_tensor<3,double>(A));

    batch_provider_new<double> bp(tree);
    if(bp.get_batchable_subspaces() != idx_list(1,0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batchable_subspaces(...) did not return correct value");
    }

    vector<idx_list> batched_subspace_grps;
    bp.get_batched_subspace_grps(batched_subspace_grps);
    vector<idx_list> correct_batched_subspace_grps(1,idx_list(1,2));
    correct_batched_subspace_grps.push_back(idx_list(1,2));
    correct_batched_subspace_grps.back().push_back(0);
    if(batched_subspace_grps != correct_batched_subspace_grps)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batched_subspace_grps(...) did not return correct value");
    }
}

} // namespace libtensor
