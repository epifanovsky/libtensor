#ifndef LIBTENSOR_OPERATOR_ADD_H
#define LIBTENSOR_OPERATOR_ADD_H

#include "add/unassigned_expression_node_add.h"
#include "ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Addition of two expressions
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator+(
    unassigned_expression<N, T> lhs,
    unassigned_expression<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n(
        new unassigned_expression_node_add<N, T>(lhs, rhs));
    return unassigned_expression<N, T>(n);
}


/** \brief Addition of a tensor and an expression
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator+(
    labeled_anytensor<N, T> lhs,
    unassigned_expression<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_lhs(
        new unassigned_expression_node_ident<N, T>(lhs));
    unassigned_expression<N, T> e_lhs(n_lhs);
    return e_lhs + rhs;
}


/** \brief Addition of an expression and a tensor
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator+(
    unassigned_expression<N, T> lhs,
    labeled_anytensor<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_rhs(
        new unassigned_expression_node_ident<N, T>(rhs));
    unassigned_expression<N, T> e_rhs(n_rhs);
    return lhs + e_rhs;
}


/** \brief Addition of two tensors
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator+(
    labeled_anytensor<N, T> lhs,
    labeled_anytensor<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_lhs(
        new unassigned_expression_node_ident<N, T>(lhs));
    std::auto_ptr< unassigned_expression_node<N, T> > n_rhs(
        new unassigned_expression_node_ident<N, T>(rhs));
    unassigned_expression<N, T> e_lhs(n_lhs);
    unassigned_expression<N, T> e_rhs(n_rhs);
    return e_lhs + e_rhs;
}


/** \brief Unary plus, returns the expression itself
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator+(
    unassigned_expression<N, T> e) {

    return e;
}


/** \brief Unary minus, returns the tensor itself
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
labeled_anytensor<N, T> operator+(
    labeled_anytensor<N, T> t) {

    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_ADD_H
