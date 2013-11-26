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

    std::multimap<size_t, size_t> map;

    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    for (size_t i = 0; i < N; i++) {
        map.insert(std::pair<size_t, size_t>(i, p[i]));
    }

    expr_tree e(expr::node_add(N, map));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
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

    std::multimap<size_t, size_t> map;
    permutation<N> p = lhs.get_label().permutation_of(rhs.get_label());
    for (size_t i = 0; i < N; i++) {
        map.insert(std::pair<size_t, size_t>(i, p[i]));
    }

    std::vector<size_t> perm(N);
    for (size_t i = 0; i < N; i++) perm[i] = i;

    expr_tree e(expr::node_add(N, map));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, lhs.get_expr());
    e.add(id, expr::node_transform<T>(perm, scalar_transf<T>(-1)));
    id = e.get_edges_out(id).back();
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
