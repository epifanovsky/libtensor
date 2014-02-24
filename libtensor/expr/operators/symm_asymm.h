#ifndef LIBTENSOR_EXPR_OPERATORS_SYMM_ASYMM_H
#define LIBTENSOR_EXPR_OPERATORS_SYMM_ASYMM_H

#include <libtensor/expr/dag/node_symm.h>

namespace libtensor {
namespace expr {


/** \brief Symmetrization of an expression over two sets of indices
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> symm(
    const label<M> sym1,
    const label<M> sym2,
    const expr_rhs<N, T> &subexpr) {

    std::vector<size_t> sym(2 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        sym[j++] = subexpr.index_of(sym1.letter_at(i));
        sym[j++] = subexpr.index_of(sym2.letter_at(i));
    }

    expr_tree e(node_symm<T>(N, sym, 2, scalar_transf<T>(),
        scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Anti-symmetrization of an expression over two sets of indices
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> asymm(
    const label<M> sym1,
    const label<M> sym2,
    const expr_rhs<N, T> &subexpr) {

    std::vector<size_t> sym(2 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        sym[j++] = subexpr.index_of(sym1.letter_at(i));
        sym[j++] = subexpr.index_of(sym2.letter_at(i));
    }

    expr_tree e(node_symm<T>(N, sym, 2, scalar_transf<T>(-1),
        scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Symmetrization of an expression over two indices
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    const expr_rhs<N, T> &subexpr) {

    return symm(label<1>(l1), label<1>(l2), subexpr);
}


/** \brief Anti-symmetrization of an expression over two indices
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    const expr_rhs<N, T> &subexpr) {

    return asymm(label<1>(l1), label<1>(l2), subexpr);
}


/** \brief Symmetrization of an expression over three sets of indices
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> symm(
    const label<M> sym1,
    const label<M> sym2,
    const label<M> sym3,
    const expr_rhs<N, T> &subexpr) {

    std::vector<size_t> sym(3 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        sym[j++] = subexpr.index_of(sym1.letter_at(i));
        sym[j++] = subexpr.index_of(sym2.letter_at(i));
        sym[j++] = subexpr.index_of(sym3.letter_at(i));
    }

    expr_tree e(node_symm<T>(N, sym, 3, scalar_transf<T>(),
        scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Anti-symmetrization of an expression over three sets of indices
    \tparam N Tensor order.
    \tparam M Number of indexes in each set.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> asymm(
    const label<M> sym1,
    const label<M> sym2,
    const label<M> sym3,
    const expr_rhs<N, T> &subexpr) {

    std::vector<size_t> sym(3 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        sym[j++] = subexpr.index_of(sym1.letter_at(i));
        sym[j++] = subexpr.index_of(sym2.letter_at(i));
        sym[j++] = subexpr.index_of(sym3.letter_at(i));
    }

    expr_tree e(node_symm<T>(N, sym, 3, scalar_transf<T>(-1),
        scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Symmetrization of an expression over three indices
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> symm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    const expr_rhs<N, T> &subexpr) {

    return symm(label<1>(l1), label<1>(l2), label<1>(l3), subexpr);
}


/** \brief Anti-symmetrization of an expression over three indices
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> asymm(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    const expr_rhs<N, T> &subexpr) {

    return asymm(label<1>(l1), label<1>(l2), label<1>(l3), subexpr);
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::symm;
using expr::asymm;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_SYMM_ASYMM_H
