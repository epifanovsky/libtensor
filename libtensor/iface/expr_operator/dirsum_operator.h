#ifndef LIBTENSOR_IFACE_DIRSUM_OPERATOR_H
#define LIBTENSOR_IFACE_DIRSUM_OPERATOR_H

#include <libtensor/expr/node_dirsum.h>

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
    const expr_rhs<N, T> &bta,
    const expr_rhs<M, T> &btb) {

    std::vector<const letter *> label(N + M);
    for (size_t i = 0; i < N; i++) {
        label[i] = bta.letter_at(i);
    }
    for (size_t i = 0, j = N; i < M; i++, j++) {
        label[j] = btb.letter_at(i);
    }

    const expr_tree &le = bta.get_expr(), &re = btb.get_expr();
    tensor_list tl(le.get_tensors());
    tl.merge(re.get_tensors());

    return expr_rhs<N + M, T>(expr_tree(expr::node_dirsum(le.get_nodes(),
            re.get_nodes()), tl), letter_expr<N + M>(label));
}


} // namespace iface

using iface::dirsum;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIRSUM_OPERATOR_H
