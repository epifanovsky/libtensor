#ifndef LIBTENSOR_TENSOR_FROM_NODE_H
#define LIBTENSOR_TENSOR_FROM_NODE_H

#include <libtensor/expr/node_ident.h>
#include "node_interm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {
using namespace libtensor::expr;


template<size_t N>
btensor<N, double> &tensor_from_node(const node &n) {

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

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

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

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
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_TENSOR_FROM_NODE_H
