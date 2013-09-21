#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_OPERATOR_H

#include "../ident/ident_core.h"
#include "add_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Addition of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr<N, T> operator+(
    expr<N, T> lhs,
    expr<N, T> rhs) {

    return expr<N, T>(new add_core<N, T>(lhs, rhs));
}


/** \brief Addition of a tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
expr<N, T> operator+(
    labeled_btensor<N, T, A1> lhs,
    expr<N, T> rhs) {

    return expr<N, T>(new ident_core<N, T, A1>(lhs)) + rhs;
}


/** \brief Addition of an expression and a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
expr<N, T> operator+(
    expr<N, T> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return lhs + expr<N, T>(new ident_core<N, T, A2>(rhs));
}


/** \brief Addition of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr<N, T> operator+(
    labeled_btensor<N, T, A1> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return
        expr<N, T>(new ident_core<N, T, A1>(lhs)) +
        expr<N, T>(new ident_core<N, T, A2>(rhs));
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator+;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_OPERATOR_H
