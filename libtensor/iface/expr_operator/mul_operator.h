#ifndef LIBTENSOR_IFACE_MUL_OPERATOR_H
#define LIBTENSOR_IFACE_MUL_OPERATOR_H

#include <libtensor/core/scalar_transform.h>
#include <libtensor/expr/node_transform.h>

namespace libtensor {
namespace iface {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const T &lhs, const expr_rhs<N, T> &rhs) {

    const expr_tree &expr = rhs.get_expr();
    return expr_rhs<N, T>(expr_tree(expr::node_transform<T>(expr.get_nodes(),
            scalar_transform<T>(lhs)), expr.get_tensors()), rhs.get_label());
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const expr_rhs<N, T> &lhs, const T &rhs) {

    const expr_tree &expr = lhs.get_expr();
    return expr_rhs<N, T>(expr_tree(expr::node_transform<T>(expr.get_nodes(),
            scalar_transform<T>(rhs)), expr.get_tensors()), lhs.get_label());
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator/(const expr_rhs<N, T> &lhs, const T &rhs) {

    const expr_tree &expr = rhs.get_expr();
    return expr_rhs<N, T>(expr_tree(expr::node_transform<T>(expr.get_nodes(),
            scalar_transform<T>(1. / rhs)), expr.get_tensors()),
            rhs.get_label());
}


} // namespace iface

using iface::operator*;
using iface::operator/;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_MUL_OPERATOR_H
