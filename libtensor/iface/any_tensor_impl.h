#ifndef LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
#define LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H

#include "any_tensor.h"
#include "expr_rhs.h"
#include <libtensor/expr/node_ident.h>

namespace libtensor {
namespace iface {


template<size_t N, typename T>
expr_rhs<N, T> any_tensor<N, T>::make_rhs(const letter_expr<N> &label) {

    return expr_rhs<N, T>(expr_tree(expr::node_ident<N, T>(*this)), label);
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_ANY_TENSOR_IMPL_H
