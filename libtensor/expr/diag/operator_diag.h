#ifndef LIBTENSOR_OPERATOR_DIAG_H
#define LIBTENSOR_OPERATOR_DIAG_H

#include "unassigned_expression_node_diag.h"
#include "../ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Extraction of a general tensor diagonal (expression)
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N - M + 1, T> diag(
    const letter &l,
    const letter_expr<M> ld,
    unassigned_expression<N, T> a) {

    std::auto_ptr< unassigned_expression_node<N - M + 1, T> > n(
        new unassigned_expression_node_diag<N, M, T>(a, ld, l));
    return unassigned_expression<N - M + 1, T>(n);
}


/** \brief Extraction of a general tensor diagonal (tensor)
    \tparam N Tensor order.
    \tparam M Diagonal order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr
 **/
template<size_t N, size_t M, typename T>
unassigned_expression<N - M + 1, T> diag(
    const letter &l,
    const letter_expr<M> ld,
    labeled_anytensor<N, T> a) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_a(
        new unassigned_expression_node_ident<N, T>(a));
    unassigned_expression<N, T> e_a(n_a);
    return diag(l, ld, e_a);
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_DIAG_H
