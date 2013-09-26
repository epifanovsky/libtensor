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
expr_rhs<N, T> operator+(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    return expr_rhs<N, T>(new add_core<N, T>(lhs, rhs));
}

#if 0
/** \brief Addition of a tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
expr_rhs<N, T> operator+(
    labeled_btensor<N, T, A1> lhs,
    expr_rhs<N, T> rhs) {

    return expr_rhs<N, T>(new ident_core<N, T, A1>(lhs)) + rhs;
}


/** \brief Addition of an expression and a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
expr_rhs<N, T> operator+(
    expr_rhs<N, T> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return lhs + expr_rhs<N, T>(new ident_core<N, T, A2>(rhs));
}


/** \brief Addition of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr_rhs<N, T> operator+(
    labeled_btensor<N, T, A1> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return
        expr_rhs<N, T>(new ident_core<N, T, A1>(lhs)) +
        expr_rhs<N, T>(new ident_core<N, T, A2>(rhs));
}
#endif


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator+;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_OPERATOR_H
