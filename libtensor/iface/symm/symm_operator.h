#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H

#include "symm2_core.h"
#include "symm3_core.h"

namespace libtensor {
namespace labeled_btensor_expr {


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
    expr_rhs<N, T> subexpr) {

    return expr_rhs<N, T>(new symm2_core<N, M, true, T>(sym1, sym2, subexpr));
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
    expr_rhs<N, T> subexpr) {

    return expr_rhs<N, T>(new symm2_core<N, M, false, T>(sym1, sym2, subexpr));
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
    expr_rhs<N, T> subexpr) {

    return expr_rhs<N, T>(new symm3_core<N, true, T>(l1, l2, l3, subexpr));
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
    expr_rhs<N, T> subexpr) {

    return expr_rhs<N, T>(new symm3_core<N, false, T>(l1, l2, l3, subexpr));
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
    expr_rhs<N, T> subexpr) {

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
    expr_rhs<N, T> subexpr) {

    return asymm(letter_expr<1>(l1), letter_expr<1>(l2), subexpr);
}


#if 0
/** \brief Symmetrization of a tensor over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
expr_rhs<N, T> symm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    labeled_btensor<N, T, A> bt) {

    return symm(sym1, sym2, expr_rhs<N, T>(new ident_core<N, T, A>(bt)));
}


/** \brief Anti-symmetrization of a tensor over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T, bool A>
expr_rhs<N, T> asymm(
    const letter_expr<M> sym1,
    const letter_expr<M> sym2,
    labeled_btensor<N, T, A> bt) {

    return asymm(sym1, sym2, expr_rhs<N, T>(new ident_core<N, T, A>(bt)));
}


/** \brief Symmetrization of a tensor over two indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    labeled_btensor<N, T, A> bt) {

    return symm(letter_expr<1>(l1), letter_expr<1>(l2), bt);
}


/** \brief Anti-symmetrization of a tensor over two indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    labeled_btensor<N, T, A> bt) {

    return asymm(letter_expr<1>(l1), letter_expr<1>(l2), bt);
}


/** \brief Symmetrization of a tensor over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    labeled_btensor<N, T, A> bt) {

    return symm(l1, l2, l3, expr_rhs<N, T>(new ident_core<N, T, A>(bt)));
}


/** \brief Anti-symmetrization of a tensor over three indexes
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam A Tensor assignable.

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T, bool A>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    labeled_btensor<N, T, A> bt) {

    return asymm(l1, l2, l3, expr_rhs<N, T>(new ident_core<N, T, A>(bt)));
}
#endif


} // namespace labeled_btensor_expr

using labeled_btensor_expr::symm;
using labeled_btensor_expr::asymm;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM_OPERATOR_H
