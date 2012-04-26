#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_EWMULT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_EWMULT_H

#include "../ident/core_ident.h"
#include "ewmult_core.h"
#include "ewmult_eval.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
expr<N + M - K, T, ewmult_core<N - K, M - K, K, T,
    expr<N, T, E1>,
    expr<M, T, E2>
> >
inline ewmult(
    const letter_expr<K> ewidx,
    expr<N, T, E1> bta,
    expr<M, T, E2> btb) {

    typedef expr<N, T, E1> expr1_t;
    typedef expr<M, T, E2> expr2_t;
    typedef ewmult_core<N - K, M - K, K, T, expr1_t, expr2_t> ewmult_t;
    return expr<N + M - K, T, ewmult_t>(ewmult_t(ewidx, bta, btb));
}


/** \brief Element-wise multiplication of two expressions

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
expr<N + M - 1, T, ewmult_core<N - 1, M - 1, 1, T,
    expr<N, T, E1>,
    expr<M, T, E2>
> >
inline ewmult(
    const letter &l,
    expr<N, T, E1> bta,
    expr<M, T, E2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of a %tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, bool A1, typename E2>
expr<N + M - K, T, ewmult_core<N - K, M - K, K, T,
    expr< N, T, core_ident<N, T, A1> >,
    expr< M, T, E2 >
> >
inline ewmult(
    const letter_expr<K> ewidx,
    labeled_btensor<N, T, A1> bta,
    expr<M, T, E2> btb) {

    typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
    return ewmult(ewidx, expr1_t(bta), btb);
}


/** \brief Element-wise multiplication of a %tensor and an expression

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, typename E2>
expr<N + M - 1, T, ewmult_core<N - 1, M - 1, 1, T,
    expr< N, T, core_ident<N, T, A1> >,
    expr< M, T, E2 >
> >
inline ewmult(
    const letter &l,
    labeled_btensor<N, T, A1> bta,
    expr<M, T, E2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of an expression and a %tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, typename E1, bool A2>
expr<N + M - K, T, ewmult_core<N - K, M - K, K, T,
    expr< N, T, E1 >,
    expr< M, T, core_ident<M, T, A2> >
> >
inline ewmult(
    const letter_expr<K> ewidx,
    expr<N, T, E1> bta,
    labeled_btensor<M, T, A2> btb) {

    typedef expr< M, T, core_ident<M, T, A2> > expr2_t;
    return ewmult(ewidx, bta, expr2_t(btb));
}


/** \brief Element-wise multiplication of an expression and a %tensor

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, typename E1, bool A2>
expr<N + M - 1, T, ewmult_core<N - 1, M - 1, 1, T,
    expr< N, T, E1 >,
    expr< M, T, core_ident<M, T, A2> >
> >
inline ewmult(
    const letter &l,
    expr<N, T, E1> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


/** \brief Element-wise multiplication of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T, bool A1, bool A2>
expr<N + M - K, T, ewmult_core<N - K, M - K, K, T,
    expr< N, T, core_ident<N, T, A1> >,
    expr< M, T, core_ident<M, T, A2> >
> >
inline ewmult(
    const letter_expr<K> ewidx,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    typedef expr< N, T, core_ident<N, T, A1> > expr1_t;
    typedef expr< M, T, core_ident<M, T, A2> > expr2_t;
    return ewmult(ewidx, expr1_t(bta), expr2_t(btb));
}


/** \brief Element-wise multiplication of two tensors

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M - 1, T, ewmult_core<N - 1, M - 1, 1, T,
    expr< N, T, core_ident<N, T, A1> >,
    expr< M, T, core_ident<M, T, A2> >
> >
inline ewmult(
    const letter &l,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return ewmult(letter_expr<1>(l), bta, btb);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::ewmult;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_OPERATOR_EWMULT_H
