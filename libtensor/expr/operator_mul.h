#ifndef LIBTENSOR_OPERATOR_MUL_H
#define LIBTENSOR_OPERATOR_MUL_H

#include "unassigned_expression.h"
#include "labeled_anytensor.h"
#include "ident/unassigned_expression_node_ident.h"
#include "scale/unassigned_expression_node_scale.h"

namespace libtensor {


/** \brief Multiplies an expression by a scalar
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> scale(
    unassigned_expression<N, T> &expr,
    const T &factor) {

    std::auto_ptr< unassigned_expression_node<N, T> > n(
        new unassigned_expression_node_scale<N, T>(expr, factor));
    return unassigned_expression<N, T>(n);
}


/** \brief Multiplication of an expression by a scalar on the left
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    const T &lhs,
    unassigned_expression<N, T> rhs) {

    return scale(rhs, lhs);
}


/** \brief Multiplication of an expression by a scalar on the right
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    unassigned_expression<N, T> lhs,
    const T &rhs) {

    return scale(lhs, rhs);
}


/** \brief Multiplication of a tensor by a scalar on the left
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    const T &lhs,
    labeled_anytensor<N, T> rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_rhs(
        new unassigned_expression_node_ident<N, T>(rhs));
    unassigned_expression<N, T> e_rhs(n_rhs);
    return scale(e_rhs, lhs);
}


/** \brief Multiplication of a tensor by a scalar on the right
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    labeled_anytensor<N, T> lhs,
    const T &rhs) {

    std::auto_ptr< unassigned_expression_node<N, T> > n_lhs(
        new unassigned_expression_node_ident<N, T>(lhs));
    unassigned_expression<N, T> e_lhs(n_lhs);
    return scale(e_lhs, rhs);
}


/** \brief Multiplication of an expression by a double on the left
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    double lhs,
    unassigned_expression<N, T> rhs) {

    return T(lhs) * rhs;
}


/** \brief Multiplication of an expression by a double on the right
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    unassigned_expression<N, T> lhs,
    double rhs) {

    return lhs * T(rhs);
}


/** \brief Multiplication of a tensor by a double on the left
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    double lhs,
    labeled_anytensor<N, T> rhs) {

    return T(lhs) * rhs;
}


/** \brief Multiplication of a tensor by a double on the right
    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
unassigned_expression<N, T> operator*(
    labeled_anytensor<N, T> lhs,
    double rhs) {

    return lhs * T(rhs);
}


/** \brief Multiplication of an expression by a double on the left
        (full specialization for double)
    \ingroup libtensor_expr
 **/
template<size_t N>
unassigned_expression<N, double> operator*(
    double lhs,
    unassigned_expression<N, double> rhs) {

    return scale(rhs, lhs);
}


/** \brief Multiplication of an expression by a double on the right
        (full specialization for double)
    \ingroup libtensor_expr
 **/
template<size_t N>
unassigned_expression<N, double> operator*(
    unassigned_expression<N, double> lhs,
    double rhs) {

    return scale(lhs, rhs);
}


/** \brief Multiplication of a tensor by a double on the left
        (full specialization for double)
    \ingroup libtensor_expr
 **/
template<size_t N>
unassigned_expression<N, double> operator*(
    double lhs,
    labeled_anytensor<N, double> rhs) {

    std::auto_ptr< unassigned_expression_node<N, double> > n_rhs(
        new unassigned_expression_node_ident<N, double>(rhs));
    unassigned_expression<N, double> e_rhs(n_rhs);
    return scale(e_rhs, lhs);
}


/** \brief Multiplication of a tensor by a double on the right
        (full specialization for double)
    \ingroup libtensor_expr
 **/
template<size_t N>
unassigned_expression<N, double> operator*(
    labeled_anytensor<N, double> lhs,
    double rhs) {

    std::auto_ptr< unassigned_expression_node<N, double> > n_lhs(
        new unassigned_expression_node_ident<N, double>(lhs));
    unassigned_expression<N, double> e_lhs(n_lhs);
    return scale(e_lhs, rhs);
}


} // namespace libtensor

#endif // LIBTENSOR_OPERATOR_MUL_H
