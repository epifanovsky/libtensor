#ifndef LIBTENSOR_OPERATOR_DIRSUM_H
#define LIBTENSOR_OPERATOR_DIRSUM_H

#include "unassigned_expression_node_dirsum.h"
#include "../ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Direct sum of two expressions
    \tparam N Order of the first expression.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M, T> dirsum(
    unassigned_expression<N, T> a,
    unassigned_expression<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N + M, T> > n(
        new unassigned_expression_node_dirsum<N, M, T>(a, b));
    return unassigned_expression<N + M, T>(n);
}


/** \brief Direct sum of a tensor and an expression
    \tparam N Order of the first tensor.
    \tparam M Order of the second expression.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M, T> dirsum(
    labeled_anytensor<N, T> a,
    unassigned_expression<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_a(
        new unassigned_expression_node_ident<N, T>(a));
    unassigned_expression<N, T> e_a(n_a);
    return dirsum(e_a, b);
}


/** \brief Direct sum of an expression and a tensor
    \tparam N Order of the first expression.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M, T> dirsum(
    unassigned_expression<N, T> a,
    labeled_anytensor<M, T> b) {

    std::auto_ptr< unassigned_expression_node<M, T> > n_b(
        new unassigned_expression_node_ident<M, T>(b));
    unassigned_expression<M, T> e_b(n_b);
    return dirsum(a, e_b);
}


/** \brief Direct sum of two tensors
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N + M, T> dirsum(
    labeled_anytensor<N, T> a,
    labeled_anytensor<M, T> b) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_a(
        new unassigned_expression_node_ident<N, T>(a));
    std::auto_ptr< unassigned_expression_node<M, T> > n_b(
        new unassigned_expression_node_ident<M, T>(b));
    unassigned_expression<N, T> e_a(n_a);
    unassigned_expression<M, T> e_b(n_b);
    return dirsum(e_a, e_b);
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_DIRSUM_H
