#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_MUL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_MUL_H

#include "core_scale.h"
#include "eval_scale.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, typename E1>
expr< N, T, core_scale< N, T, expr<N, T, E1> > >
inline operator/(
    expr<N, T, E1> lhs,
    T rhs) {

    typedef expr<N, T, E1> expr1_t;
    typedef core_scale<N, T, expr1_t> scale_expr_t;
    typedef expr<N, T, scale_expr_t> expr_t;
    return expr_t(scale_expr_t(1. / rhs, lhs));
}


/** \brief Division of a tensor by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr< N, T, core_scale< N, T, expr< N, T, core_ident<N, T, A> > > >
inline operator/(
    labeled_btensor<N, T, A> lhs,
    T rhs) {

    typedef expr< N, T, core_ident<N, T, A> > expr1_t;
    return expr1_t(lhs) / rhs;
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator/;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_MUL_H
