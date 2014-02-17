#ifndef LIBTENSOR_EXPR_OPERATORS_DIRSUM_H
#define LIBTENSOR_EXPR_OPERATORS_DIRSUM_H

#include <libtensor/expr/dag/node_dirsum.h>

namespace libtensor {
namespace expr {


/** \brief Direct sum of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M, T> dirsum(
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    std::vector<const letter *> lab(N + M);
    for(size_t i = 0; i < N; i++) {
        lab[i] = &a.letter_at(i);
    }
    for(size_t i = 0, j = N; i < M; i++, j++) {
        lab[j] = &b.letter_at(i);
    }

    // TODO: remap tensors

    expr_tree e(node_dirsum(N + M));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, a.get_expr());
    e.add(id, b.get_expr());
    return expr_rhs<N + M, T>(e, label<N + M>(lab));
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::dirsum;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_DIRSUM_H
