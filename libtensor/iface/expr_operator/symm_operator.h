#ifndef LIBTENSOR_IFACE_SYMM_OPERATOR_H
#define LIBTENSOR_IFACE_SYMM_OPERATOR_H

#include "symm2_core.h"
#include "symm3_core.h"

namespace libtensor {
namespace iface {


/** \brief Symmetrization of an expression over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> symm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    const expr_rhs<N, T> &subexpr) {

    return expr_rhs<N, T>(new symm2_core<N, M, T>(sym1, sym2, subexpr));
}


/** \brief Anti-symmetrization of an expression over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> asymm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    const expr_rhs<N, T> &subexpr) {

    return expr_rhs<N, T>(new symm2_core<N, M, T>(sym1, sym2, subexpr));
}


/** \brief Symmetrization of an expression over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    const expr_rhs<N, T> &subexpr) {

    return expr_rhs<N, T>(new symm3_core<N, T>(l1, l2, l3, subexpr));
}


/** \brief Anti-symmetrization of an expression over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    const expr_rhs<N, T> &subexpr) {

    return expr_rhs<N, T>(new symm3_core<N, T>(l1, l2, l3, subexpr));
}


/** \brief Symmetrization of an expression over two indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    const expr_rhs<N, T> &subexpr) {

    return symm(letter_expr<1>(l1), letter_expr<1>(l2), subexpr);
}


/** \brief Anti-symmetrization of an expression over two indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    const expr_rhs<N, T> &subexpr) {

    return asymm(letter_expr<1>(l1), letter_expr<1>(l2), subexpr);
}


} // namespace iface

using iface::symm;
using iface::asymm;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_SYMM_OPERATOR_H
