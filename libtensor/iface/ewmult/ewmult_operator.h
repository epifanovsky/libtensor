#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_OPERATOR_H

#include "ewmult_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> ewidx,
    expr_rhs<N, T> bta,
    expr_rhs<M, T> btb) {

    typedef ewmult_core<N - K, M - K, K, T> ewmult_core_t;
    return expr_rhs<N + M - K, T>(new ewmult_core_t(ewidx, bta, btb));
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    expr_rhs<N, T> bta,
    expr_rhs<M, T> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of a tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, bool A1>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> ewidx,
    labeled_btensor<N, T, A1> bta,
    expr_rhs<M, T> btb) {

    return ewmult(ewidx, expr_rhs<N, T>(new ident_core<N, T, A1>(bta)), btb);
}


/** \brief Element-wise multiplication of a tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    labeled_btensor<N, T, A1> bta,
    expr_rhs<M, T> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of an expression and a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, bool A2>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> ewidx,
    expr_rhs<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(ewidx, bta, expr_rhs<M, T>(new ident_core<M, T, A2>(btb)));
}


/** \brief Element-wise multiplication of an expression and a tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A2>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    expr_rhs<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, bool A1, bool A2>
expr_rhs<N + M - K, T> ewmult(
    const letter_expr<K> ewidx,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(
        ewidx,
        expr_rhs<N, T>(new ident_core<N, T, A1>(bta)),
        expr_rhs<M, T>(new ident_core<M, T, A2>(btb)));
}


/** \brief Element-wise multiplication of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr_rhs<N + M - 1, T> ewmult(
    const letter &l,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::ewmult;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_OPERATOR_H
