#ifndef LIBTENSOR_OPERATOR_CONTRACT_H
#define LIBTENSOR_OPERATOR_CONTRACT_H

#include "unassigned_expression_node_contract2.h"
#include "../ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Contraction of two expressions over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first expression.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T>
unassigned_expression<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    unassigned_expression<N, T> a,
    unassigned_expression<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N + M - 2 * K, T> > n(
        new unassigned_expression_node_contract2<N - K, M - K, K, T>(
            contr, a, b));
    return unassigned_expression<N + M - 2 * K, T>(n);
}


/** \brief Contraction of two expressions over one index
    \tparam N Order of the first expression.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M - 2, T> contract(
    const letter &let,
    unassigned_expression<N, T> a,
    unassigned_expression<M, T> b) {

    return contract(letter_expr<1>(let), a, b);
}


/** \brief Contraction of a tensor and an expression over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first tensor.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T>
unassigned_expression<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    labeled_anytensor<N, T> a,
    unassigned_expression<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_a(
        new unassigned_expression_node_ident<N, T>(a));
    unassigned_expression<N, T> e_a(n_a);
    return contract(contr, e_a, b);
}


/** \brief Contraction of a %tensor and an expression over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M - 2, T> contract(
    const letter &let,
    labeled_anytensor<N, T> a,
    unassigned_expression<M, T> b) {

    return contract(letter_expr<1>(let), a, b);
}


/** \brief Contraction of an expression and a tensor over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first expression.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T>
unassigned_expression<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    unassigned_expression<N, T> a,
    labeled_anytensor<M, T> b) {

    std::auto_ptr< unassigned_expression_node<M, T> > n_b(
        new unassigned_expression_node_ident<M, T>(b));
    unassigned_expression<M, T> e_b(n_b);
    return contract(contr, a, e_b);
}


/** \brief Contraction of an expression and a tensor over one index
    \tparam N Order of the first expression.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M - 2, T> contract(
    const letter &let,
    unassigned_expression<N, T> a,
    labeled_anytensor<M, T> b) {

    return contract(letter_expr<1>(let), a, b);
}


/** \brief Contraction of two tensors over multiple indexes
    \tparam K Number of contracted indexes.
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t K, size_t N, size_t M, typename T>
unassigned_expression<N + M - 2 * K, T> contract(
    const letter_expr<K> contr,
    labeled_anytensor<N, T> a,
    labeled_anytensor<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_a(
        new unassigned_expression_node_ident<N, T>(a));
    std::auto_ptr< unassigned_expression_node<M, T> > n_b(
        new unassigned_expression_node_ident<M, T>(b));
    unassigned_expression<N, T> e_a(n_a);
    unassigned_expression<M, T> e_b(n_b);
    return contract(contr, e_a, e_b);
}


/** \brief Contraction of two tensors over one index
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M - 2, T> contract(
    const letter &let,
    labeled_anytensor<N, T> a,
    labeled_anytensor<M, T> b) {

    return contract(letter_expr<1>(let), a, b);
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_CONTRACT_H
