#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H

#include "scale_core.h"
#include "eval_scale.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E2>
expr< N, T, scale_core< N, T, expr<N, T, E2> > >
inline operator*(
    T lhs,
    expr<N, T, E2> rhs) {

    typedef expr<N, T, E2> expr2_t;
    typedef scale_core<N, T, expr2_t> scale_expr_t;
    typedef expr<N, T, scale_expr_t> expr_t;
    return expr_t(scale_expr_t(lhs, rhs));
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1>
expr< N, T, scale_core< N, T, expr<N, T, E1> > >
inline operator*(
    expr<N, T, E1> lhs,
    T rhs) {

    typedef expr<N, T, E1> expr1_t;
    typedef scale_core<N, T, expr1_t> scale_expr_t;
    typedef expr<N, T, scale_expr_t> expr_t;
    return expr_t(scale_expr_t(rhs, lhs));
}


/** \brief Multiplication of a tensor by a scalar from the left

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr< N, T, scale_core< N, T, expr< N, T, core_ident<N, T, A> > > >
inline operator*(
    T lhs,
    labeled_btensor<N, T, A> rhs) {

    typedef expr< N, T, core_ident<N, T, A> > expr2_t;
    return lhs * expr2_t(rhs);
}


/** \brief Multiplication of a tensor by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr< N, T, scale_core< N, T, expr< N, T, core_ident<N, T, A> > > >
inline operator*(
    labeled_btensor<N, T, A> lhs,
    T rhs) {

    typedef expr< N, T, core_ident<N, T, A> > expr1_t;
    return expr1_t(lhs) * rhs;
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1>
expr< N, T, scale_core< N, T, expr<N, T, E1> > >
inline operator/(
    expr<N, T, E1> lhs,
    T rhs) {

    typedef expr<N, T, E1> expr1_t;
    typedef scale_core<N, T, expr1_t> scale_expr_t;
    typedef expr<N, T, scale_expr_t> expr_t;
    return expr_t(scale_expr_t(1. / rhs, lhs));
}


/** \brief Division of a tensor by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr< N, T, scale_core< N, T, expr< N, T, core_ident<N, T, A> > > >
inline operator/(
    labeled_btensor<N, T, A> lhs,
    T rhs) {

    typedef expr< N, T, core_ident<N, T, A> > expr1_t;
    return expr1_t(lhs) / rhs;
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator*;
using labeled_btensor_expr::operator/;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MUL_OPERATOR_H
