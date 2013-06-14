#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H

#include "scale_core.h"
#include "../ident/ident_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator*(
    T lhs,
    expr<N, T> rhs) {

    return expr<N, T>(scale_core<N, T>(lhs, rhs));
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator*(
    expr<N, T> lhs,
    T rhs) {

    return expr<N, T>(scale_core<N, T>(rhs, lhs));
}


/** \brief Multiplication of a tensor by a scalar from the left

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr<N, T> operator*(
    T lhs,
    labeled_btensor<N, T, A> rhs) {

    return lhs * expr<N, T>(ident_core<N, T, A>(rhs));
}


/** \brief Multiplication of a tensor by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr<N, T> operator*(
    labeled_btensor<N, T, A> lhs,
    T rhs) {

    return expr<N, T>(ident_core<N, T, A>(lhs)) * rhs;
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator/(
    expr<N, T> lhs,
    T rhs) {

    return expr<N, T>(scale_core<N, T>(1.0 / rhs, lhs));
}


/** \brief Division of a tensor by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr<N, T> operator/(
    labeled_btensor<N, T, A> lhs,
    T rhs) {

    return expr<N, T>(ident_core<N, T, A>(lhs)) / rhs;
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator*;
using labeled_btensor_expr::operator/;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H
