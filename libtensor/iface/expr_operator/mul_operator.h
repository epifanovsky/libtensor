#ifndef LIBTENSOR_IFACE_MUL_OPERATOR_H
#define LIBTENSOR_IFACE_MUL_OPERATOR_H

#include <libtensor/core/scalar_transf.h>
#include <libtensor/expr/node_transform.h>

namespace libtensor {
namespace iface {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const T &lhs, const expr_rhs<N, T> &rhs) {

    std::vector<size_t> perm(N);
    for(size_t i = 0; i < N; i++) perm[i] = i;

    expr::expr_tree e(expr::node_transform<T>(perm, scalar_transf<T>(lhs)));
    e.add(e.get_root(), rhs.get_expr());
    return expr_rhs<N, T>(e, rhs.get_label());
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const expr_rhs<N, T> &lhs, const T &rhs) {

    return rhs * lhs;
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator/(const expr_rhs<N, T> &lhs, const T &rhs) {

    return (1/rhs) * lhs;
}


} // namespace iface

using iface::operator*;
using iface::operator/;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_MUL_OPERATOR_H
