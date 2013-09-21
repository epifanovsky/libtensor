#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H

#include "../ident/ident_core.h"
#include "diag_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Extraction of a general tensor diagonal (expression)
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr<N - M + 1, T> diag(
    const letter &let_diag,
    const letter_expr<M> &lab_diag,
    const expr<N, T> &subexpr) {

    return expr<N - M + 1, T>(new diag_core<N, M, T>(let_diag,
            lab_diag, subexpr));
}


/** \brief Extraction of a general tensor diagonal (tensor)
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
expr<N - M + 1, T> diag(
    const letter &let_diag,
    const letter_expr<M> &lab_diag,
    const labeled_btensor<N, T, A> bt) {

    return diag(let_diag, lab_diag, expr<N, T>(new ident_core<N, T, A>(bt)));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::diag;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIAG_OPERATOR_H
