#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_OPERATOR_H

#include "../ident/ident_core.h"
#include "dirsum_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Direct sum of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M, T> dirsum(
    expr_rhs<N, T> bta,
    expr_rhs<M, T> btb) {

    return expr_rhs<N + M, T>(new dirsum_core<N, M, T>(bta, btb));
}


/** \brief Direct sum of a tensor and an expression
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1>
expr_rhs<N + M, T> dirsum(
    labeled_btensor<N, T, A1> bta,
    expr_rhs<M, T> btb) {

    return dirsum(expr_rhs<N, T>(new ident_core<N, T, A1>(bta)), btb);
}


/** \brief Direct sum of an expression and a tensor
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A2>
expr_rhs<N + M, T> dirsum(
    expr_rhs<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return dirsum(bta, expr_rhs<M, T>(new ident_core<M, T, A2>(btb)));
}


/** \brief Direct sum of two tensors
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr_rhs<N + M, T> dirsum(
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return dirsum(
        expr_rhs<N, T>(new ident_core<N, T, A1>(bta)),
        expr_rhs<M, T>(new ident_core<M, T, A2>(btb)));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::dirsum;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_OPERATOR_H
