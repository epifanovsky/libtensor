#ifndef LIBTENSOR_IFACE_SYMM_OPERATOR_H
#define LIBTENSOR_IFACE_SYMM_OPERATOR_H

#include <libtensor/expr/dag/node_symm.h>

namespace libtensor {
namespace iface {


/** \brief Symmetrization of an expression over two sets of indices
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

    std::vector<size_t> sym(2 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        const letter &l1 = sym1.letter_at(i);
        const letter &l2 = sym2.letter_at(i);

        sym[j++] = subexpr.index_of(l1);
        sym[j++] = subexpr.index_of(l2);
    }

    expr::expr_tree e(expr::node_symm<T>(N, sym, 2,
            scalar_transf<T>(), scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Anti-symmetrization of an expression over two sets of indices
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

    std::vector<size_t> sym(2 * M, 0);
    for(size_t i = 0, j = 0; i < M; i++) {
        const letter &l1 = sym1.letter_at(i);
        const letter &l2 = sym2.letter_at(i);

        sym[j++] = subexpr.index_of(l1);
        sym[j++] = subexpr.index_of(l2);
    }

    expr::expr_tree e(expr::node_symm<T>(N, sym, 2,
            scalar_transf<T>(-1), scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Symmetrization of an expression over three indices
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

    std::vector<size_t> sym(3, 0);
    sym[0] = subexpr.index_of(l1);
    sym[1] = subexpr.index_of(l2);
    sym[2] = subexpr.index_of(l3);

    expr::expr_tree e(expr::node_symm<T>(N, sym, 3,
            scalar_transf<T>(), scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
}


/** \brief Anti-symmetrization of an expression over three indices
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

    std::vector<size_t> sym(3, 0);
    sym[0] = subexpr.index_of(l1);
    sym[1] = subexpr.index_of(l2);
    sym[2] = subexpr.index_of(l3);

    expr::expr_tree e(expr::node_symm<T>(N, sym, 3,
            scalar_transf<T>(-1), scalar_transf<T>()));
    e.add(e.get_root(), subexpr.get_expr());
    return expr_rhs<N, T>(e, subexpr.get_label());
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
