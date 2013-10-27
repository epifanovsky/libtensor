#ifndef LIBTENSOR_IFACE_DIAG_OPERATOR_H
#define LIBTENSOR_IFACE_DIAG_OPERATOR_H

#include <libtensor/expr/node_diag.h>

namespace libtensor {
namespace iface {


/** \brief Extraction of a general tensor diagonal (expression)
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N - M + 1, T> diag(
    const letter &let_diag,
    const letter_expr<M> &lab_diag,
    const expr_rhs<N, T> &subexpr) {

    std::vector<size_t> diagdims(M, 0);
    std::vector<const letter *> label;
    for (size_t i = 0, j = 0; i < N; i++) {
        const letter &l = bta.letter_at(i);
        if (l == let_diag || ! lab_diag.contains(l)) label.push_back(&l);
        else diagdims[lab_diag.index_of(l)] = i;
    }
    if (label.size() != N - M + 1) {
        throw expr_exception(g_ns, "", "diag(const letter &, "
                "const letter_expr<M> &, expr_rhs<N, T> &)",
                __FILE__, __LINE__, "Error in letters");
    }

    const expr_tree &ex = subexpr.get_expr();
    return expr_rhs<N - M + 1, T>(expr_tree(expr::node_diag(ex.get_nodes(),
            diagdims), ex.get_tensors()), letter_expr<N - M + 1>(label));
}


} // namespace iface

using iface::diag;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIAG_OPERATOR_H
