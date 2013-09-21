#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_OPERATOR_H

#include "../ident/ident_core.h"
#include "direct_product_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Direct product of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr<N + M, T> operator*(
    expr<N, T> bta,
    expr<M, T> btb) {

    return expr<N + M, T>(new direct_product_core<N, M, T>(bta, btb));
}


/** \brief Direct product of a tensor and an expression
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1>
expr<N + M, T> operator*(
    labeled_btensor<N, T, A1> bta,
    expr<M, T> btb) {

    return expr<N, T>(new ident_core<N, T, A1>(bta)) * btb;
}


/** \brief Direct product of an expression and a %ensor
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A2>
expr<N + M, T> operator*(
    expr<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return bta * expr<M, T>(new ident_core<M, T, A2>(btb));
}


/** \brief Direct product of two tensors
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M, T> operator*(
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return expr<N, T>(new ident_core<N, T, A1>(bta)) *
        expr<M, T>(new ident_core<M, T, A2>(btb));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator*;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_PRODUCT_OPERATOR_H
