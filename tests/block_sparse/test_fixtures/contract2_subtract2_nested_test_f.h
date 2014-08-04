#ifndef CONTRACT2_SUBTRACT2_NESTED_TEST_F_H
#define CONTRACT2_SUBTRACT2_NESTED_TEST_F_H

#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor.h>

#include <libtensor/expr/dag/expr_tree.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_transform.h>

#include "util.h"
#include "contract2_test_f.h"

namespace libtensor {

// C = AB
// G = C - F
// E = D G^T
class contract2_subtract2_nested_test_f  : public contract2_test_f
{
private:
    static const double s_F_arr[18];
    static const double s_D_arr[21]; 
    static const double s_E_arr[18];

    static const size_t ml_sparsity[4][2];

    static sparse_bispace<1> init_m();


public:
    sparse_bispace<1> spb_m;
    sparse_bispace<2> spb_D;
    sparse_bispace<2> spb_E;

    sparse_btensor<3> A;
    sparse_btensor<3> B;
    direct_sparse_btensor<2> C;
    sparse_btensor<2> F;
    direct_sparse_btensor<2> G;
    sparse_btensor<2> D;
    sparse_btensor<2> E;
    sparse_btensor<2> E_correct;

    expr::expr_tree tree; 


    contract2_subtract2_nested_test_f() : spb_m(init_m()),
                                          spb_D(spb_m % spb_l << get_sig_blocks(ml_sparsity,4)),
                                          spb_E(spb_m | spb_i),
                                          A(spb_A,s_A_arr,true),
                                          B(spb_B,s_B_arr,true),
                                          C(spb_C),
                                          F(spb_C,s_F_arr,true),
                                          G(spb_C),
                                          D(spb_D,s_D_arr,true),
                                          E(spb_E),
                                          E_correct(spb_E,s_E_arr,true),
                                          tree(expr::node_assign(2))

                                          
    {
        using namespace expr;
        //Build expression tree for this set of operations
        expr_tree::node_id_t root_id = tree.get_root();
        tree.add(root_id, node_ident_any_tensor<2,double>(E));

        //The numbers UP toward the  leaves of the tree (the earliest steps)
        //Confusing I know
        std::multimap<size_t,size_t> contr_map_0;
        contr_map_0.insert(idx_pair(1,3));
        node_contract contr_node_0(2,contr_map_0,true);
        expr_tree::node_id_t contr_node_id_0 = tree.add(root_id,contr_node_0);
        tree.add(contr_node_id_0,node_ident_any_tensor<2,double>(D));


        node_assign interm_assign_node_0(2);
        expr_tree::node_id_t interm_assign_node_id_0 = tree.add(contr_node_id_0,interm_assign_node_0);
        tree.add(interm_assign_node_id_0,node_ident_any_tensor<2,double>(G));

        //Subtraction node LHS
        node_add n_add(2); 
        expr_tree::node_id_t n_add_id = tree.add(interm_assign_node_id_0,n_add);

        node_assign interm_assign_node_1(2);
        expr_tree::node_id_t interm_assign_node_id_1 = tree.add(n_add_id,interm_assign_node_1);
        tree.add(interm_assign_node_id_1,node_ident_any_tensor<2,double>(C));

        std::multimap<size_t,size_t> contr_map_1;
        contr_map_1.insert(idx_pair(1,3));
        contr_map_1.insert(idx_pair(2,4));
        node_contract contr_node_1(2,contr_map_1,true);
        expr_tree::node_id_t contr_node_id_1 = tree.add(interm_assign_node_id_1,contr_node_1);
        tree.add(contr_node_id_1,node_ident_any_tensor<3,double>(A));
        tree.add(contr_node_id_1,node_ident_any_tensor<3,double>(B));

        //Subtraction node RHS
        idx_list perm_entries(1,0);
        perm_entries.push_back(1);
        node_transform<double> transf_node(perm_entries,scalar_transf<double>(-1));
        expr_tree::node_id_t n_transf_id = tree.add(n_add_id,transf_node);

        tree.add(n_transf_id,node_ident_any_tensor<2,double>(F));
    }
};

} // namespace libtensor



#endif /* CONTRACT2_SUBTRACT2_NESTED_TEST_F_H */
