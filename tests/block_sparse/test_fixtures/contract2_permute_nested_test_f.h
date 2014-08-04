#ifndef CONTRACT2_PERMUTE_NESTED_TEST_F_H
#define CONTRACT2_PERMUTE_NESTED_TEST_F_H

#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/block_sparse/direct_sparse_btensor_new.h>

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

class contract2_permute_nested_test_f  : public contract2_test_f
{
private:
    static const double s_D_arr[18];
public:
    sparse_bispace<2> spb_D;
    sparse_btensor<3> A;
    sparse_btensor<3> B;
    direct_sparse_btensor<2> C;
    sparse_btensor<2> D;
    sparse_btensor<2> D_correct;

    expr::expr_tree tree;

    contract2_permute_nested_test_f() : spb_D(spb_l | spb_i),
                                        A(spb_A,s_A_arr,true),
                                        B(spb_B,s_B_arr,true),
                                        C(spb_C),
                                        D(spb_D),
                                        D_correct(spb_D,s_D_arr,true),
                                        tree(expr::node_assign(2))


    {
        using namespace expr;
        expr_tree::node_id_t root_id = tree.get_root();
        tree.add(root_id, node_ident_any_tensor<2,double>(D));

        idx_list perm_entries(1,1);
        perm_entries.push_back(0);
        node_transform<double> perm_node(perm_entries, scalar_transf<double>());
        expr_tree::node_id_t perm_node_id = tree.add(root_id,perm_node);


        node_assign interm_assign_node(2);
        expr_tree::node_id_t interm_assign_node_id = tree.add(perm_node_id,interm_assign_node);
        tree.add(interm_assign_node_id,node_ident_any_tensor<2,double>(C));


        std::multimap<size_t,size_t> contr_map;
        contr_map.insert(idx_pair(1,3));
        contr_map.insert(idx_pair(2,4));
        node_contract contr_node(2,contr_map,true);
        expr_tree::node_id_t contr_node_id = tree.add(interm_assign_node_id,contr_node);

        tree.add(contr_node_id,node_ident_any_tensor<3,double>(A));
        tree.add(contr_node_id,node_ident_any_tensor<3,double>(B));
    }
};



} // namespace libtensor


#endif /* CONTRACT2_PERMUTE_NESTED_TEST_F_H */
