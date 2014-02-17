#ifndef LIBTENSOR_EXPR_OPERATORS_PLUS_MINUS_H
#define LIBTENSOR_EXPR_OPERATORS_PLUS_MINUS_H

#include <libtensor/core/scalar_transf.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_transform.h>
#include "multiply_divide.h"

namespace libtensor {
namespace expr {


/** \brief Addition of two expressions

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator+(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    node_add add(N);
    expr_tree e(add);
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    if (! p.is_identity()) {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = p[i];

        id = e.add(id, node_transform<T>(perm, scalar_transf<T>()));
    }
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Subtraction of an expression from an expression

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    node_add add(N);
    expr_tree e(add);
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    std::vector<size_t> perm(N);
    for (size_t i = 0; i < N; i++) perm[i] = p[i];

    id = e.add(id, node_transform<T>(perm, scalar_transf<T>(-1)));
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Do nothing to an expression

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
const expr_rhs<N, T> &operator+(
    const expr_rhs<N, T> &rhs) {

    return rhs;
}


/** \brief Change of the sign of an expression

    \ingroup libtensor_expr_operators
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &rhs) {

    return T(-1) * rhs;
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::operator+;
using expr::operator-;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_OPERATORS_PLUS_MINUS_H
