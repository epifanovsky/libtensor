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
    sparse_btensor_new<2> C(tf.spb_C);

    node_assign root(2);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    e.add(root_id, node_ident_any_tensor<2,double>(C));

    multimap<size_t,size_t> contr_map;
    contr_map.insert(idx_pair(1,3));
    contr_map.insert(idx_pair(2,4));
    node_contract contr_node(2,contr_map,true);
    expr_tree::node_id_t contr_node_id = e.add(root_id,contr_node);
    e.add(contr_node_id,node_ident_any_tensor<3,double>(A));
    e.add(contr_node_id,node_ident_any_tensor<3,double>(B));

    //Finally, check that we can get the full tensor
    batch_provider_new<double> bp(e);
    bp.get_batch((double*)C.get_data_ptr());

    sparse_btensor_new<2> C_correct(tf.spb_C,tf.C_arr,true);

    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value contract2 test case");
    }
}

void batch_provider_test::test_contract2_permute_nested() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_contract2_permute_nested()";
    contract2_permute_nested_test_f tf; 

    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<2> C(tf.spb_C);
    sparse_btensor_new<2> D(tf.spb_D);

    node_assign root(2);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    e.add(root_id, node_ident_any_tensor<2,double>(D));

    idx_list perm_entries(1,1);
    perm_entries.push_back(0);
    node_transform<double> perm_node(perm_entries, scalar_transf<double>());
    expr_tree::node_id_t perm_node_id = e.add(root_id,perm_node);


    node_assign interm_assign_node(2);
    expr_tree::node_id_t interm_assign_node_id = e.add(perm_node_id,interm_assign_node);
    e.add(interm_assign_node_id,node_ident_any_tensor<2,double>(C));


    multimap<size_t,size_t> contr_map;
    contr_map.insert(idx_pair(1,3));
    contr_map.insert(idx_pair(2,4));
    node_contract contr_node(2,contr_map,true);
    expr_tree::node_id_t contr_node_id = e.add(interm_assign_node_id,contr_node);

    e.add(contr_node_id,node_ident_any_tensor<3,double>(A));
    e.add(contr_node_id,node_ident_any_tensor<3,double>(B));

    batch_provider_new<double> bp(e);
    bp.get_batch((double*)D.get_data_ptr());


    sparse_btensor_new<2> D_correct(tf.spb_D,tf.D_arr,true);
    if(D != D_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for contract2_permute_nested test case");
    }
}

void batch_provider_test::test_contract2_subtract2_nested() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_contract2_subtract2_nested()";
    contract2_subtract2_nested_test_f tf; 

    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<2> C(tf.spb_C);
    sparse_btensor_new<2> F(tf.spb_C,tf.F_arr,true);
    direct_sparse_btensor_new<2> G(tf.spb_C);
    sparse_btensor_new<2> D(tf.spb_D,tf.D_arr,true);
    sparse_btensor_new<2> E(tf.spb_E);

    node_assign root(2);
    expr_tree e(root);
    expr_tree::node_id_t root_id = e.get_root();
    e.add(root_id, node_ident_any_tensor<2,double>(E));

    //The numbers UP toward the  leaves fo the tree (the earliest steps)
    //Confusing I know
    multimap<size_t,size_t> contr_map_0;
    contr_map_0.insert(idx_pair(1,3));
    node_contract contr_node_0(2,contr_map_0,true);
    expr_tree::node_id_t contr_node_id_0 = e.add(root_id,contr_node_0);
    e.add(contr_node_id_0,node_ident_any_tensor<2,double>(D));


    node_assign interm_assign_node_0(2);
    expr_tree::node_id_t interm_assign_node_id_0 = e.add(contr_node_id_0,interm_assign_node_0);
    e.add(interm_assign_node_id_0,node_ident_any_tensor<2,double>(G));

    //Subtraction node LHS
    node_add n_add(2); 
    expr_tree::node_id_t n_add_id = e.add(interm_assign_node_id_0,n_add);

    node_assign interm_assign_node_1(2);
    expr_tree::node_id_t interm_assign_node_id_1 = e.add(n_add_id,interm_assign_node_1);
    e.add(interm_assign_node_id_1,node_ident_any_tensor<2,double>(C));

    multimap<size_t,size_t> contr_map_1;
    contr_map_1.insert(idx_pair(1,3));
    contr_map_1.insert(idx_pair(2,4));
    node_contract contr_node_1(2,contr_map_1,true);
    expr_tree::node_id_t contr_node_id_1 = e.add(interm_assign_node_id_1,contr_node_1);
    e.add(contr_node_id_1,node_ident_any_tensor<3,double>(A));
    e.add(contr_node_id_1,node_ident_any_tensor<3,double>(B));

    //Subtraction node RHS
    idx_list perm_entries(1,0);
    perm_entries.push_back(1);
    node_transform<double> transf_node(perm_entries,scalar_transf<double>(-1));
    expr_tree::node_id_t n_transf_id = e.add(n_add_id,transf_node);

    e.add(n_transf_id,node_ident_any_tensor<2,double>(F));

    batch_provider_new<double> bp(e);
    bp.get_batch((double*)E.get_data_ptr());


    sparse_btensor_new<2> E_correct(tf.spb_E,tf.E_arr,true);
    if(E != E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_provider::get_batch(...) did not return correct value for contract2_subtract2_nested test case");
    }
}

} // namespace libtensor
