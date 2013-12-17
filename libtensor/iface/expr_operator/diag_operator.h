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

    enum {
        NC = N - M + 1
    };

    std::vector<size_t> diagdims(M, 0);
    std::vector<const letter*> label;
    bool diag_added = false;
    for(size_t i = 0, j = 0; i < N; i++) {
        const letter &l = subexpr.letter_at(i);
        if(!lab_diag.contains(l)) {
            label.push_back(&l);
        } else {
            if(!diag_added) {
                label.push_back(&let_diag);
                diag_added = true;
            }
            diagdims[lab_diag.index_of(l)] = i;
        }
    }
    if(label.size() != NC) {
        throw expr_exception(g_ns, "", "diag(const letter &, "
                "const letter_expr<M> &, expr_rhs<N, T> &)",
                __FILE__, __LINE__, "Error in letters");
    }

    expr::expr_tree e(expr::node_diag(NC, diagdims));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<NC, T>(e, letter_expr<NC>(label));
}


} // namespace iface

using iface::diag;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_DIAG_OPERATOR_H
