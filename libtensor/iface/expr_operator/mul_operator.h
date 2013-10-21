#ifndef LIBTENSOR_IFACE_MUL_OPERATOR_H
#define LIBTENSOR_IFACE_MUL_OPERATOR_H

#include "../expr_core/scale_core.h"

namespace libtensor {
namespace iface {


/** \brief Multiplication of an expression by a scalar from the left

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const T &lhs, const expr_rhs<N, T> &rhs) {

    expr_core_ptr<N, T> core(new scale_core<N, T>(lhs, rhs.get_core()));
    return expr_rhs<N, T>(core, rhs.get_label());
}


/** \brief Multiplication of an expression by a scalar from the right

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator*(const expr_rhs<N, T> &lhs, const T &rhs) {

    expr_core_ptr<N, T> core(new scale_core<N, T>(rhs, lhs.get_core()));
    return expr_rhs<N, T>(core, lhs.get_label());
}


/** \brief Division of an expression by a scalar

    \ingroup libtensor_btensor_expr_op
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator/(const expr_rhs<N, T> &lhs, const T &rhs) {

    expr_core_ptr<N, T> core(new scale_core<N, T>(1. / rhs, lhs.get_core()));
    return expr_rhs<N, T>(core, lhs.get_label());
}


} // namespace iface

using iface::operator*;
using iface::operator/;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_MUL_OPERATOR_H
