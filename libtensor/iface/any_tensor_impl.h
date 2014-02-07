#ifndef LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
#define LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H

#include "any_tensor.h"
#include "expr_rhs.h"
#include <libtensor/expr/node_ident_any_tensor.h>

namespace libtensor {
namespace iface {


template<size_t N, typename T>
expr_rhs<N, T> any_tensor<N, T>::make_rhs(const letter_expr<N> &label) {

    expr::expr_tree e(expr::node_ident_any_tensor<N, T>(*this));
    return expr_rhs<N, T>(e, label);
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
