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
expr_rhs<N, T> operator*(
    const T &lhs,
    const expr_rhs<N, T> &rhs) {

    return expr_rhs<N, T>(new scale_core<N, T>(lhs, rhs));
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(
    const expr_rhs<N, T> &lhs,
    const T &rhs) {

    return expr_rhs<N, T>(new scale_core<N, T>(rhs, lhs));
}


#if 0
/** \brief Multiplication of a tensor by a scalar from the left

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> operator*(
    const T &lhs,
    labeled_btensor<N, T, A> rhs) {

    return lhs * expr_rhs<N, T>(new ident_core<N, T, A>(rhs));
}


/** \brief Multiplication of a tensor by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> operator*(
    labeled_btensor<N, T, A> lhs,
    const T &rhs) {

    return expr_rhs<N, T>(new ident_core<N, T, A>(lhs)) * rhs;
}
#endif


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator/(
    const expr_rhs<N, T> &lhs,
    const T &rhs) {

    return expr_rhs<N, T>(new scale_core<N, T>(1.0 / rhs, lhs));
}


#if 0
/** \brief Division of a tensor by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> operator/(
    labeled_btensor<N, T, A> lhs,
    const T &rhs) {

    return expr_rhs<N, T>(new ident_core<N, T, A>(lhs)) / rhs;
}
#endif


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator*;
using labeled_btensor_expr::operator/;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H
