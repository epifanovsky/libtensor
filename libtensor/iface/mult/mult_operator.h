#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_OPERATOR_H

#include "../ident/ident_core.h"
#include "mult_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> mult(
    expr<N, T> lhs,
    expr<N, T> rhs) {

    return expr<N, T>(mult_core<N, T, false>(lhs, rhs));
}


/** \brief Element-wise multiplication of a tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
expr<N, T> mult(
    labeled_btensor<N, T, A1> lhs,
    expr<N, T> rhs) {

    return mult(expr<N, T>(ident_core<N, T, A1>(lhs)), rhs);
}


/** \brief Element-wise multiplication of an expression and a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
expr<N, T> mult(
    expr<N, T> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return mult(lhs, expr<N, T>(ident_core<N, T>(rhs)));
}


/** \brief Element-wise multiplication of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T> mult(
    labeled_btensor<N, T, A1> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return mult(
        expr<N, T>(ident_core<N, T, A1>(lhs)),
        expr<N, T>(ident_core<N, T, A2>(rhs)));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::mult;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_OPERATOR_H
