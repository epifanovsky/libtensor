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
expr_rhs<N, T> operator-(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    return lhs + expr_rhs<N, T>(new scale_core<N, T>(T(-1), rhs));
}

#if 0
/** \brief Subtraction of a tensor from an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A2>
expr_rhs<N, T> operator-(
    expr_rhs<N, T> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return lhs - expr_rhs<N, T>(new ident_core<N, T, A2>(rhs));
}


/** \brief Subtraction of an expression from a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1>
expr_rhs<N, T> operator-(
    labeled_btensor<N, T, A1> lhs,
    expr_rhs<N, T> rhs) {

    return expr_rhs<N, T>(new ident_core<N, T, A1>(lhs)) - rhs;
}


/** \brief Subtraction of a tensor from a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A1, bool A2>
expr_rhs<N, T> operator-(
    labeled_btensor<N, T, A1> lhs,
    labeled_btensor<N, T, A2> rhs) {

    return
        expr_rhs<N, T>(new ident_core<N, T, A1>(lhs)) -
        expr_rhs<N, T>(new ident_core<N, T, A2>(rhs));
}
#endif


/** \brief Unary minus (expression), effectively multiplies by -1

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    expr_rhs<N, T> e) {

    return expr_rhs<N, T>(new scale_core<N, T>(T(-1), e));
}


#if 0
/** \brief Unary minus on a tensor, effectively multiplies by -1

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> operator-(
    labeled_btensor<N, T, A> t) {

    return -expr_rhs<N, T>(new ident_core<N, T, A>(t));
}
#endif


} // namespace labeled_btensor_expr

using labeled_btensor_expr::operator-;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SUB_OPERATOR_H
