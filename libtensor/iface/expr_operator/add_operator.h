#ifndef LIBTENSOR_IFACE_ADD_OPERATOR_H
#define LIBTENSOR_IFACE_ADD_OPERATOR_H

#include "../expr_core/add_core.h"
#include "../expr_core/scale_core.h"

namespace libtensor {
namespace iface {


/** \brief Addition of two expressions

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator+(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    expr_core_ptr<N, T> core(new add_core<N, T>(lhs.get_core(),
            rhs.get_core(), match(lhs.get_label(), rhs.get_label())));
    return expr_rhs<N, T>(core, lhs.get_label());
}


/** \brief Subtraction of an expression from an expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
expr_rhs<N, T> operator-(
    expr_rhs<N, T> lhs,
    expr_rhs<N, T> rhs) {

    expr_core_ptr<N, T> core(new scale_core<N, T>(T(-1), rhs.get_core()));
    return lhs + expr_rhs<N, T>(core, rhs.get_label());
}


} // namespace iface

using iface::operator+;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_ADD_OPERATOR_H
