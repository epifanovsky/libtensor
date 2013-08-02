#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_OPERATOR_H

#include "../ident/ident_core.h"
#include "contract2_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Contraction of two expressions over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first %tensor.
    \tparam M Order of the second %tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T>
expr<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    expr<N, T> bta,
    expr<M, T> btb) {

    typedef contract2_core<N - K, M - K, K, T> core_t;
    typedef expr<N + M - 2 * K, T> expr_t;
    return expr_t(core_t(contr, bta, btb));
}


/** \brief Contraction of two expressions over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr<N + M - 2, T> contract(
    const letter &let,
    expr<N, T> bta,
    expr<M, T> btb) {

    return contract(letter_expr<1>(let), bta, btb);
}


/** \brief Contraction of a tensor and an expression over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T, bool A1>
expr<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    labeled_btensor<N, T, A1> bta,
    expr<M, T> btb) {

    return contract(contr, expr<N, T>(ident_core<N, T, A1>(bta)), btb);
}


/** \brief Contraction of a tensor and an expression over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1>
expr<N + M - 2, T> contract(
    const letter &let,
    labeled_btensor<N, T, A1> bta,
    expr<M, T> btb) {

    return contract(letter_expr<1>(let), bta, btb);
}


/** \brief Contraction of an expression and a tensor over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T, bool A2>
expr<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    expr<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return contract(contr, bta, expr<M, T>(ident_core<M, T, A2>(btb)));
}


/** \brief Contraction of an expression and a tensor over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A2>
expr<N + M - 2, T> contract(
    const letter &let,
    expr<N, T> bta,
    labeled_btensor<M, T, A2> btb) {

    return contract(letter_expr<1>(let), bta, btb);
}


/** \brief Contraction of two tensors over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t K, size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return contract(contr, expr<N, T>(ident_core<N, T, A1>(bta)),
        expr<M, T>(ident_core<M, T, A2>(btb)));
}


/** \brief Contraction of two tensors over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.
    \tparam A1 First tensor assignable.
    \tparam A2 Second tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A1, bool A2>
expr<N + M - 2, T> contract(
    const letter &let,
    labeled_btensor<N, T, A1> bta,
    labeled_btensor<M, T, A2> btb) {

    return contract(letter_expr<1>(let), bta, btb);
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::contract;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_OPERATOR_H
