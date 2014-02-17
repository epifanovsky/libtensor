#ifndef LIBTENSOR_EXPR_TENSOR_FROM_NODE_H
#define LIBTENSOR_EXPR_TENSOR_FROM_NODE_H

#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include "node_interm.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N, typename T>
class btensor_from_node {
private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_head; //!< Head node ID
    tensor_transf<N, T> m_tr; //!< Transformation from leaf to head
    expr_tree::node_id_t m_leaf; //!< Leaf node ID

public:
    btensor_from_node(const expr_tree &tree, expr_tree::node_id_t head);

    const tensor_transf<N, T> &get_transf() const {
        return m_tr;
    }

    btensor<N, T> &get_btensor() const;
    btensor<N, T> &get_or_create_btensor(const block_index_space<N> &bis);

private:
    static expr_tree::node_id_t inspect_node(const expr_tree &tree,
        expr_tree::node_id_t head, tensor_transf<N, T> &tr);

};


template<size_t N, typename T>
btensor_from_node<N, T>::btensor_from_node(const expr_tree &tree,
    expr_tree::node_id_t head) :
    m_tree(tree), m_head(head),
    m_leaf(inspect_node(tree, head, m_tr)) {

}


template<size_t N, typename T>
btensor<N, T> &btensor_from_node<N, T>::get_btensor() const {

    const node &n = m_tree.get_vertex(m_leaf);

    if(n.get_op().compare(node_ident::k_op_type) == 0) {

        const node_ident_any_tensor<N, double> &ni =
            n.recast_as< node_ident_any_tensor<N, double> >();
        return btensor<N, double>::from_any_tensor(ni.get_tensor());

    } else if(n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
            n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());
        if(ph.is_empty()) throw 75;
        return ph.get_btensor();

    } else {
        throw 76;
    }
}


template<size_t N, typename T>
btensor<N, T> &btensor_from_node<N, T>::get_or_create_btensor(
    const block_index_space<N> &bis) {

    const node &n = m_tree.get_vertex(m_leaf);

    if(n.get_op().compare(node_ident::k_op_type) == 0) {

        const node_ident_any_tensor<N, double> &ni =
            n.recast_as< node_ident_any_tensor<N, double> >();
        return btensor<N, double>::from_any_tensor(ni.get_tensor());

    } else if(n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
            n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());
        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();

    } else {
        throw 77;
    }
}


template<size_t N, typename T>
expr_tree::node_id_t btensor_from_node<N, T>::inspect_node(
    const expr_tree &tree, expr_tree::node_id_t head, tensor_transf<N, T> &tr) {

    const node &n = tree.get_vertex(head);

    if(n.get_op().compare(node_ident::k_op_type) == 0) {
        return head;
    } else if(n.get_op().compare(node_interm_base::k_op_type) == 0) {
        return head;
    } else if(n.get_op().compare(node_transform_base::k_op_type) == 0) {
        const node_transform<T> &nt = n.recast_as< node_transform<T> >();
        sequence<N, size_t> seq1(0), seq2(0);
        for(size_t i = 0; i < N; i++) {
            seq1[i] = i;
            seq2[i] = nt.get_perm().at(i);
        }
        permutation_builder<N> pb(seq2, seq1);
        tensor_transf<N, T> tr1(pb.get_perm(), nt.get_coeff());

        const expr_tree::edge_list_t &e = tree.get_edges_out(head);
        expr_tree::node_id_t leaf = inspect_node(tree, e[0], tr);
        tr.transform(tr1);
        return leaf;
    } else {
        throw 74;
    }
}

template<size_t N>
btensor<N, double> &tensor_from_node(const node &n) {

    if (n.get_op().compare(node_ident::k_op_type) == 0) {
        const node_ident_any_tensor<N, double> &ni =
                n.recast_as< node_ident_any_tensor<N, double> >();

        return btensor<N, double>::from_any_tensor(ni.get_tensor());
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) throw 73;
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


template<size_t N>
btensor<N, double> &tensor_from_node(const node &n,
    const block_index_space<N> &bis) {

    if (n.get_op().compare(node_ident::k_op_type) == 0) {
        const node_ident_any_tensor<N, double> &ni =
                n.recast_as< node_ident_any_tensor<N, double> >();

        return btensor<N, double>::from_any_tensor(ni.get_tensor());
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_TENSOR_FROM_NODE_H
