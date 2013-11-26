#ifndef LIBTENSOR_IFACE_DIRSUM_OPERATOR_H
#define LIBTENSOR_IFACE_DIRSUM_OPERATOR_H

#include <libtensor/expr/node_add.h>

namespace libtensor {
namespace iface {


/** \brief Direct sum of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M, T> dirsum(
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    std::multimap<size_t, size_t> map;

    std::vector<const letter *> label(N + M);
    for(size_t i = 0; i < N; i++) {
        label[i] = &a.letter_at(i);
    }
    for(size_t i = 0, j = N; i < M; i++, j++) {
        label[j] = &b.letter_at(i);
    }

    // TODO: remap tensors

    expr::expr_tree e(expr::node_add(N, map));
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, a.get_expr());
    e.add(id, b.get_expr());
    return expr_rhs<N + M, T>(e, letter_expr<N + M>(label));
}


} // namespace iface

using iface::dirsum;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIRSUM_OPERATOR_H
