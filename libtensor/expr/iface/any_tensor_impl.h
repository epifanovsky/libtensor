#ifndef LIBTENSOR_EXPR_ANY_TENSOR_IMPL_H
#define LIBTENSOR_EXPR_ANY_TENSOR_IMPL_H

#include "any_tensor.h"
#include "expr_rhs.h"
#include "node_ident_any_tensor.h"

namespace libtensor {
namespace expr {


template<size_t N, typename T>
expr_rhs<N, T> any_tensor<N, T>::make_rhs(const label<N> &l) {

    expr_tree e(node_ident_any_tensor<N, T>(*this));
    return expr_rhs<N, T>(e, l);
}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_ANY_TENSOR_IMPL_H
