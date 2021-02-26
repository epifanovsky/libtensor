#ifndef LIBTENSOR_EXPR_OPERATORS_MULTIPLY_DIVIDE_H
#define LIBTENSOR_EXPR_OPERATORS_MULTIPLY_DIVIDE_H

#include <libtensor/core/scalar_transf.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_transform.h>

namespace libtensor {
namespace expr {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const double &lhs, const expr_rhs<N, T> &rhs) {

    std::vector<size_t> perm(N);
    for(size_t i = 0; i < N; i++) perm[i] = i;

    expr_tree e(node_transform<T>(perm, scalar_transf<T>(lhs)));
    e.add(e.get_root(), rhs.get_expr());
    return expr_rhs<N, T>(e, rhs.get_label());
}

template<size_t N, typename T>
expr_rhs<N, T> operator*(const float &lhs, const expr_rhs<N, T> &rhs) {

    std::vector<size_t> perm(N);
    for(size_t i = 0; i < N; i++) perm[i] = i;

    expr_tree e(node_transform<T>(perm, scalar_transf<T>(lhs)));
    e.add(e.get_root(), rhs.get_expr());
    return expr_rhs<N, T>(e, rhs.get_label());
}

/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const expr_rhs<N, T> &lhs, const T &rhs) {

    return rhs * lhs;
}


/** \brief Direct product of two expressions
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.
    \tparam T Tensor element type.

    \ingroup libtensor_expr_operators
 **/
template<size_t N, size_t M, typename T>
expr_rhs<N + M, T> operator*(
    const expr_rhs<N, T> &a,
    const expr_rhs<M, T> &b) {

    std::multimap<size_t, size_t> cseq;
    std::vector<const letter*> lab(N + M);
    for(size_t i = 0; i < N; i++) {
        lab[i] = &a.letter_at(i);
    }
    for(size_t i = 0, j = N; i < M; i++, j++) {
        lab[j] = &b.letter_at(i);
    }

    expr_tree e(node_contract(N + M, cseq, true));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, a.get_expr());
    e.add(id, b.get_expr());

    return expr_rhs<N + M, T>(e, label<N + M>(lab));
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator/(const expr_rhs<N, T> &lhs, const T &rhs) {

    return (T(1) / rhs) * lhs;
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::operator*;
using expr::operator/;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_MULTIPLY_DIVIDE_H
