#ifndef LIBTENSOR_EXPR_OPERATORS_DIAG_H
#define LIBTENSOR_EXPR_OPERATORS_DIAG_H

#include <libtensor/expr/dag/node_diag.h>

namespace libtensor {
namespace expr {


/** \brief Extraction of a general tensor diagonal
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N - M + 1, T> diag(
    const letter &let_diag,
    const label<M> &lab_diag,
    const expr_rhs<N, T> &subexpr) {

    enum {
        NC = N - M + 1
    };

    std::vector<size_t> idx(N, 0), oidx(N - M + 1, 0);
    for(size_t i = 0; i < N; i++) {
        const letter &l = subexpr.letter_at(i);
        if(!lab_diag.contains(l)) {
            idx[i] = i + 1;
        } else {
            idx[i] = 0;
        }
    }

    node_diag ndiag(NC, idx, 0);
    ndiag.build_output_indices(oidx);

    std::vector<const letter*> lab(N - M + 1, 0);
    for(size_t i = 0; i < N - M + 1; i++) {
        if(oidx[i] == 0) {
            lab[i] = &let_diag;
        } else {
            lab[i] = &subexpr.letter_at(oidx[i] - 1);
        }
    }

    expr_tree e(ndiag);
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<NC, T>(e, label<NC>(lab));
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::diag;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_DIAG_H
