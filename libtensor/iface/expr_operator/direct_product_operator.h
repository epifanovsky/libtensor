#ifndef LIBTENSOR_IFACE_DIRECT_PRODUCT_OPERATOR_H
#define LIBTENSOR_IFACE_DIRECT_PRODUCT_OPERATOR_H

#include <libtensor/expr/node_dirprod.h>

namespace libtensor {
namespace iface {


/** \brief Direct product of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M, T> operator*(
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    const expr_tree &ea = a.get_expr(), &eb = b.get_expr();
    tensor_list tl(ea.get_tensors());
    tl.merge(eb.get_tensors());

    // TODO: remap tensors

    std::vector<const letter*> label(N + M);
    for(size_t i = 0; i < N; i++) {
        label[i] = &a.letter_at(i);
    }
    for(size_t i = 0, j = N; i < M; i++, j++) {
        label[j] = &b.letter_at(i);
    }

    expr::node_dirprod ndp(ea.get_nodes(), eb.get_nodes());
    return expr_rhs<N + M, T>(expr_tree(ndp, tl), letter_expr<N + M>(label));
}


} // namespace iface

using iface::operator*;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIRECT_PRODUCT_OPERATOR_H
