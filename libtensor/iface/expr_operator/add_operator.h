#ifndef LIBTENSOR_IFACE_ADD_OPERATOR_H
#define LIBTENSOR_IFACE_ADD_OPERATOR_H

#include <libtensor/core/scalar_transf.h>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_transform.h>
#include "mul_operator.h"

namespace libtensor {
namespace iface {


/** \brief Addition of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator+(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    expr::node_add add(N);
    expr::expr_tree e(add);
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    if (! p.is_identity()) {
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) perm[i] = p[i];

        id = e.add(id, expr::node_transform<T>(perm, scalar_transf<T>()));
    }
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Subtraction of an expression from an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &lhs,
    const expr_rhs<N, T> &rhs) {

    expr::node_add add(N);
    expr::expr_tree e(add);
    expr::expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    std::vector<size_t> perm(N);
    for (size_t i = 0; i < N; i++) perm[i] = p[i];

    id = e.add(id, expr::node_transform<T>(perm, scalar_transf<T>(-1)));
    e.add(id, rhs.get_expr());

    return expr_rhs<N, T>(e, lhs.get_label());
}


/** \brief Do nothing to an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
const expr_rhs<N, T> &operator+(
    const expr_rhs<N, T> &rhs) {

    return rhs;
}


/** \brief Change of the sign of an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    const expr_rhs<N, T> &rhs) {

    return T(-1) * rhs;
}


} // namespace iface

using iface::operator+;
using iface::operator-;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_ADD_OPERATOR_H
