#ifndef LIBTENSOR_EXPR_OPERATORS_SET_H
#define LIBTENSOR_EXPR_OPERATORS_SET_H

#include <libtensor/expr/dag/node_scalar.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/expr_exception.h>

namespace libtensor {
namespace expr {


/** \brief Copy tensor structure of expr with all elements set to value
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \param expr Tensor expression to copy structure from
    \param value Value to set all elements to

    Example:
    \code
    c(i|j) = set(1.0, a(i|j));
    \endcode

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> set(T &value, const expr_rhs<N, T> &expr) {

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; i++) idx[i] = i;

    node_set n(idx);
    expr_tree e(n);
    e.add(e.get_root(), expr.get_expr());
    e.add(e.get_root(), node_scalar<T>(value));

    return expr_rhs<N, T>(e, expr.get_label());
}


/** \brief Set diagonal elements of expr to a value
    \tparam N Tensor order.
    \tparam K Diagonal order.
    \tparam M Full diagonal order.
    \tparam T Tensor element type.
    \param l1 Label of diagonal indices
    \param l2 Label of indices forming the diagonal
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    w(i|a|j|k|b) = set(i|a, i|j|k|a|b, e, v(i|a|j|k|b));
    \endcode
    performs the mathematical operation
    \f[
    w_{iajkb} = v_{iajkb} + (e - v_{iaiia}) \delta_{ij}\delta_{jk}\delta_{ab}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t K, size_t M, typename T>
expr_rhs<N, T> set(const label<K> &l1, const label<M> &l2,
    T &value, const expr_rhs<N, T> &expr) {

    static const char method[] = "set(const label<K> &, "
            "const label<M> &, const T &, const expr_rhs<NA, T> &)";

    if (! l1.contains(l2.letter_at(0))) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "First letter in l2 not found in l1.");
    }

    size_t j = 0;

    std::vector<size_t> d(M, 0);
    for (size_t i = 1, nd = 1; i < M; i++, nd++) {
        if (l1.contains(l2.letter_at(i))) {
            if (nd == 1) {
                throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                        "Only one index in diagonal.");
            }
            nd = 0;
            j++;
        }
        d[i] = j;
    }
    if (j + 1 != K) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Inconsistency in l1 and l2.");
    }

    j = 0;

    std::vector<size_t> idx(N, 0);
    for(size_t i = 0, k = 0; i < N; i++) {
        const letter &l = expr.letter_at(i);
        if(! l2.contains(l)) {
            idx[i] = K + k;
            k++;
        }
        else {
            idx[i] = d[l2.index_of(l)];
            j++;
        }
    }
    if (j != M) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Unknown index in l2.");
    }

    node_set n(idx);
    expr_tree e(n);
    e.add(e.get_root(), expr.get_expr());
    e.add(e.get_root(), node_scalar<T>(value));
    return expr_rhs<N, T>(e, expr.get_label());
}


/** \brief Set diagonal elements of expr to a value
    \tparam N Tensor order.
    \tparam M Total diagonal order
    \tparam T Tensor element type.
    \param l1 Label of diagonal indices
    \param l2 Label of indices forming the diagonal
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    c(i|j|k|a) = set(i, i|j|k, e, a(i|j|k|a));
    \endcode
    \f[
    c_{ijka} = a_{ijka} + (e - A_{iiia}) \delta_{ij}\delta_{jk}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> set(const letter &l1, const label<M> &l2, T &value,
    const expr_rhs<N, T> &expr) {

    return set(label<1>(l1), l2, value, expr);
}


/** \brief Set diagonal elements of expr to a value
    \tparam N Tensor order.
    \tparam M Total diagonal order
    \tparam T Tensor element type.
    \param l Label of diagonal
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    c(i|j|k|a) = set(i|j|k, e, a(i|j|k|a));
    \endcode
    \f[
    c_{ijka} = a_{ijka} + (e - A_{iiia}) \delta_{ij}\delta_{jk}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> set(const label<M> &l, T &value,
    const expr_rhs<N, T> &expr) {

    return set(label<1>(l.letter_at(0)), l, value, expr);
}


/** \brief Shift diagonal of expr by a value
    \tparam N Tensor order.
    \tparam M Diagonal order
    \tparam K Total diagonal order
    \tparam M Diagonal order.
    \tparam T Tensor element type.
    \param l1 Label
    \param l2 Label of diagonal letters
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    c(i|a|j|k|b) = shift(i|a, i|j|k|a|b, e, a(i|a|j|k|b));
    \endcode
    \f[
    c_{iajkb} = a_{iajkb} + e \delta_{ij}\delta_{jk}\delta_{ab}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, size_t K, typename T>
expr_rhs<N, T> shift(const label<K> &l1, const label<M> &l2, T &value,
    const expr_rhs<N, T> &expr) {

    static const char method[] = "shift(const label<K> &, "
            "const label<M> &, const T &, const expr_rhs<NA, T> &)";

    if (! l1.contains(l2.letter_at(0))) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "First letter in l2 not found in l1.");
    }

    size_t j = 0;

    std::vector<size_t> d(M, 0);
    for (size_t i = 1, nd = 1; i < M; i++, nd++) {
        if (l1.contains(l2.letter_at(i))) {
            if (nd == 1) {
                throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                        "Only one index in diagonal.");
            }
            nd = 0;
            j++;
        }
        d[i] = j;
    }
    if (j + 1 != K) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Inconsistency in l1 and l2.");
    }

    j = 0;

    std::vector<size_t> idx(N, 0);
    for(size_t i = 0, k = 0; i < N; i++) {
        const letter &l = expr.letter_at(i);
        if(! l2.contains(l)) {
            idx[i] = K + k;
            k++;
        }
        else {
            idx[i] = d[l2.index_of(l)];
            j++;
        }
    }
    if (j != M) {
        throw expr_exception(g_ns, "", method, __FILE__, __LINE__,
                "Unknown index in l2.");
    }

    node_set n(idx, true);
    expr_tree e(n);
    e.add(e.get_root(), expr.get_expr());
    e.add(e.get_root(), node_scalar<T>(value));
    return expr_rhs<N, T>(e, expr.get_label());
}


/** \brief Shift diagonal elements of expr by a value
    \tparam N Tensor order.
    \tparam M Total diagonal order
    \tparam T Tensor element type.
    \param l1 Label of diagonal indices
    \param l2 Label of indices forming the diagonal
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    c(i|j|k|a) = shift(i, i|j|k, e, a(i|j|k|a));
    \endcode
    \f[
    c_{ijka} = a_{ijka} + e \delta_{ij}\delta_{jk}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> shift(const letter &l1, const label<M> &l2, T &value,
    const expr_rhs<N, T> &expr) {

    return shift(label<1>(l1), l2, value, expr);
}


/** \brief Shift diagonal of expr by a value
    \tparam N Tensor order.
    \tparam M Total diagonal order.
    \tparam T Tensor element type.
    \param l Label of diagonal letters
    \param value Value to set the diagonal elements
    \param expr Tensor expression

    Example:
    \code
    C(i|a|j|k) = shift(i|j|k, E, A(i|j|k|a));
    \endcode
    \f[
    C_{iajk} = A_{ijka} + E \delta_{ij}\delta_{jk}
    \f]

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N, T> shift(const label<M> &l, T &value,
    const expr_rhs<N, T> &expr) {

    return shift(label<1>(l.letter_at(0)), l, value, expr);

}


} // namespace expr

} // namespace libtensor


namespace libtensor {

using expr::diag;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_DIAG_H
