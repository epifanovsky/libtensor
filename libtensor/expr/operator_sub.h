#ifndef LIBTENSOR_OPERATOR_SUB_H
#define LIBTENSOR_OPERATOR_SUB_H

#include "operator_add.h"
#include "operator_mul.h"
#include "ident/unassigned_expression_node_ident.h"

namespace libtensor {


/** \brief Subtraction of an expression from an expression
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    unassigned_expression<N, T> lhs,
    unassigned_expression<N, T> rhs) {

    return lhs + rhs * T(-1);
}


/** \brief Subtraction of a tensor from an expression
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    unassigned_expression<N, T> lhs,
    labeled_anytensor<N, T> rhs) {

    return lhs + rhs * T(-1);
}


/** \brief Subtraction of an expression from a tensor
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    labeled_anytensor<N, T> lhs,
    unassigned_expression<N, T> rhs) {

    return lhs + rhs * T(-1);
}


/** \brief Subtraction of a tensor from a tensor
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    labeled_anytensor<N, T> lhs,
    labeled_anytensor<N, T> rhs) {

    return lhs + rhs * T(-1);
}


/** \brief Unary minus, multiplication of an expression by -1
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    unassigned_expression<N, T> e) {

    return scale(e, T(-1));
}


/** \brief Unary minus, multiplication of a tensor by -1
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator-(
    labeled_anytensor<N, T> t) {

    std::auto_ptr< unassigned_expression_node<N, T> > n(
        new unassigned_expression_node_ident<N, T>(t));
    unassigned_expression<N, T> e(n);
    return scale(e, T(-1));
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_SUB_H
