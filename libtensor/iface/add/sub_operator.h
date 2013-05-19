#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SUB_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SUB_OPERATOR_H

#include "../scale/scale_core.h"
#include "add_operator.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Subtraction of an expression from an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator-(
    expr<N, T> lhs,
    expr<N, T> rhs) {

    return lhs + expr<N, T>(scale_core<N, T>(T(-1), rhs));
}


/** \brief Subtraction of a tensor from an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
expr<N, T> operator-(
    expr<N, T> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return lhs - expr<N, T>(ident_core<N, T, A2>(rhs));
}


/** \brief Subtraction of an expression from a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
expr<N, T> operator-(
    labeled_btensor<N, T, A1> lhs,
    expr<N, T> rhs) {

    return expr<N, T>(ident_core<N, T, A1(lhs)) - rhs;
}


/** \brief Subtraction of a tensor from a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T> operator-(
    labeled_btensor<N, T, A1> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return expr<N, T>(ident_core<N, T, A1>(lhs)) -
        expr<N, T>(ident_core<N, T, A2>(rhs));
}


/** \brief Unary minus (expression), effectively multiplies by -1

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator-(
    expr<N, T> e) {

    return expr<N, T>(scale_core<N, T>(T(-1), e));
}


/** \brief Unary minus on a tensor, effectively multiplies by -1

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr<N, T> operator-(
    labeled_btensor<N, T, A> t) {

    return -expr<N, T>(ident_core<N, T, A>(t));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator-;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SUB_OPERATOR_H
