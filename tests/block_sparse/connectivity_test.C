#include "connectivity_test.h"
#include "test_fixtures/contract2_subtract2_nested_test_f.h"
#include "test_fixtures/contract2_permute_nested_test_f.h"
#include <libtensor/block_sparse/connectivity.h>



namespace libtensor {

using namespace expr;
using namespace std;

template<size_t N>
bool verify_conn(const connectivity& conn,size_t (&correct_arr)[N])
{
    size_t m = 0;
    for(size_t tensor_idx = 0; tensor_idx < conn.get_n_tensors(); ++tensor_idx)
    {
        size_t order = conn.get_order(tensor_idx);
        for(size_t subspace_idx = 0; subspace_idx < order; ++subspace_idx)
        {
            size_t n_conn_subspaces = conn.get_n_conn_subspaces(tensor_idx,subspace_idx);
            for(size_t conn_subspace_idx = 0; conn_subspace_idx < n_conn_subspaces ; ++conn_subspace_idx)
            {
                size_t conn_val_0 = conn(tensor_idx,subspace_idx,conn_subspace_idx).first;
                size_t correct_val_0 = correct_arr[m++];
                size_t conn_val_1 = conn(tensor_idx,subspace_idx,conn_subspace_idx).second;
                size_t correct_val_1 = correct_arr[m++];
                if((conn_val_0 != correct_val_0) || (conn_val_1 != correct_val_1)) 
                {
                    return false;
                }
            }
        }
    }
    if(m != N)
    {
        return false;
    }
    return true;
}

void connectivity_test::perform() throw(libtest::test_exception)
{
    test_addition();
    test_contract2();
    test_permute();
}

void connectivity_test::test_addition() throw(libtest::test_exception)
{
    static const char *test_name = "connectivity_test::test_addition()";

    contract2_subtract2_nested_test_f tf;
    expr_tree& tree = tf.tree;
    expr_tree::edge_list_t assign_0_children = tree.get_edges_out(tree.get_root());
    expr_tree::edge_list_t contract_0_children = tree.get_edges_out(assign_0_children[1]);

    expr_tree add_sub_tree = tree.get_subtree(contract_0_children[1]);
    connectivity conn_add(add_sub_tree);

    size_t add_correct_arr[24] = { //tensor 0 subspace 0
                                   1,0, 
                                   2,0,


                                   //tensor 0 subspace 1
                                   1,1,
                                   2,1,

                                   //tensor 1 subspace 0
                                   0,0,
                                   2,0,

                                   //tensor 1 subspace 1
                                   0,1,
                                   2,1,

                                   //tensor 2 subspace 0
                                   0,0,
                                   1,0,

                                   //tensor 2 subspace 1
                                   0,1,
                                   1,1};


    if(!verify_conn(conn_add,add_correct_arr))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "connectivity::operator(...) did not return correct value for addition test case");
    }
}

void connectivity_test::test_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "connectivity_test::test_contract2()";

    contract2_subtract2_nested_test_f tf;
    expr_tree& tree = tf.tree;
    connectivity conn_contr_0(tree);

    size_t conn_contr_0_correct_arr[12] = { //tensor 0 subspace 0
                                            1,0,

                                            //tensor 0 subspace 1
                                            2,0,

                                            //tensor 1 subspace 0
                                            0,0,

                                            //tensor 1 subspace 1
                                            2,1,

                                            //tensor 2 subspace 0
                                            0,1,

                                            //tensor 2 subspace 1
                                            1,1};

    if(!verify_conn(conn_contr_0,conn_contr_0_correct_arr))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "connectivity::operator(...) did not return correct value for contract test case 0");
    }


    expr_tree::edge_list_t assign_0_children = tree.get_edges_out(tree.get_root());
    expr_tree::edge_list_t contract_0_children = tree.get_edges_out(assign_0_children[1]);
    expr_tree::edge_list_t interm_assign_0_children = tree.get_edges_out(contract_0_children[1]);
    expr_tree::edge_list_t addition_children = tree.get_edges_out(interm_assign_0_children[1]);
    connectivity conn_contr_1(tree.get_subtree(addition_children[0]));

    size_t conn_contr_1_correct_arr[16] = { //tensor 0 subspace 0
                                            1,0,

                                            //tensor 0 subspace 1
                                            2,2,

                                            //tensor 1 subspace 0
                                            0,0,

                                            //tensor 1 subspace 1
                                            2,0,

                                            //tensor 1 subspace 2
                                            2,1,

                                            //tensor 2 subspace 0 
                                            1,1,

                                            //tensor 2 subspace 1
                                            1,2,

                                            //tensor 2 subspace 2
                                            0,1};

    if(!verify_conn(conn_contr_1,conn_contr_1_correct_arr))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "connectivity::operator(...) did not return correct value for contract test case 1");
    }
}

void connectivity_test::test_permute() throw(libtest::test_exception)
{
    static const char *test_name = "connectivity_test::test_permute()";

    contract2_permute_nested_test_f tf;
    connectivity conn_perm(tf.tree);

    size_t conn_perm_correct_arr[8] = { //tensor 0 subspace 0
                                        1,1,

                                        //tensor 0 subspace 1
                                        1,0,

                                        //tensor 1 subspace 0
                                        0,1,

                                        //tensor 1 subspace 1
                                        0,0};

                                         

    if(!verify_conn(conn_perm,conn_perm_correct_arr))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "connectivity::operator(...) did not return correct value for permutation test case");
    }
}

} // namespace libtensor
